"""Inference repair utilities for unified option-scoring proxy models."""

import torch
import torch.nn.functional as F


def proxy_inference(
    logits_sel,
    inventory,
    demand,
    threshold=0.5,
    repair=True,
    eligibility_mask=None,
    stochastic=False,
    top_k=5,
    num_dcs=None,
    num_carriers=None,
    debug=False,
    repair_strategy="argmax_then_split",
    inventory_weight_power=1.0,
):
    """
    Decode unified logits into a concrete fulfillment plan.

    Args
    -------
    logits_sel : tensor [B, D*C] - unified (DC, Carrier) logits
    inventory  : tensor [B, D] - available inventory per DC (required for repair)
    demand     : tensor [B, 1] - order quantity
    threshold  : float - unused for unified mode (kept for API compatibility)
    repair     : bool - enable greedy splitting across DCs when primary DC lacks inventory
    eligibility_mask : [B, D, C] or [B, D*C] - binary mask for eligible (DC, Carrier) pairs
    stochastic : bool - enable top-k sampling instead of argmax
    top_k      : int - number of top candidates to sample from in stochastic mode
    num_dcs    : int - number of DCs
    num_carriers : int - number of carriers
    debug      : bool - log feasibility/eligibility fallbacks
    repair_strategy : str - selection strategy for primary DC
        (argmax_then_split/default, inventory_first, feasible_topk, inventory_weighted, feasible_joint_topk)
    inventory_weight_power : float - exponent on inventory weighting for inventory_weighted strategy

    Returns
    --------
    (plan_before [B, D], plan_after [B, D], carrier_per_dc [B, D])
    """
    if num_dcs is None or num_carriers is None:
        raise ValueError("Unified proxy_inference requires num_dcs and num_carriers")

    B = logits_sel.size(0)
    D, C = num_dcs, num_carriers
    expected_size = D * C
    if logits_sel.size(1) != expected_size:
        raise ValueError(f"Unified proxy_inference expected logits of shape [B, {expected_size}] but got {logits_sel.shape}")

    logits_grid = logits_sel.view(B, D, C)

    if eligibility_mask is not None:
        elig_grid = eligibility_mask.view(B, D, C) if eligibility_mask.ndim == 2 else eligibility_mask
    else:
        elig_grid = torch.ones(B, D, C, device=logits_sel.device)

    has_eligible = (elig_grid.view(B, -1).sum(dim=1) > 0)
    if debug and (~has_eligible).any():
        missing = (~has_eligible).nonzero(as_tuple=True)[0].tolist()
        print(f"[proxy][inference] No eligible options for batch indices: {missing}")

    # Mask ineligible options
    logits_masked = logits_grid.clone()
    logits_masked[elig_grid == 0] = float('-inf')

    # Softmax over flattened options; fallback to uniform if all masked
    logits_flat = logits_masked.view(B, -1)
    if (~has_eligible).any():
        logits_flat = torch.where(has_eligible.unsqueeze(1), logits_flat, torch.zeros_like(logits_flat))

    probs_flat = F.softmax(logits_flat, dim=1)
    probs_grid = probs_flat.view(B, D, C)

    def select_carrier_for_dc(batch_idx, dc_idx):
        probs_c = probs_grid[batch_idx, dc_idx]
        elig_c = elig_grid[batch_idx, dc_idx] if elig_grid is not None else None
        if elig_c is not None:
            masked = probs_c * elig_c
            if masked.sum() > 1e-9:
                probs_c = masked / (masked.sum() + 1e-9)
            else:
                elig_idx = (elig_c > 0).nonzero(as_tuple=True)[0]
                return int(elig_idx[0].item()) if len(elig_idx) > 0 else 0
        if stochastic:
            top_probs_c, top_idx_c = torch.topk(probs_c, k=min(top_k, C), dim=0)
            top_probs_c = top_probs_c / (top_probs_c.sum() + 1e-9)
            choice = torch.multinomial(top_probs_c, num_samples=1).item()
            return int(top_idx_c[choice].item())
        return int(probs_c.argmax().item())

    primary_dc = None
    primary_carrier = None

    if (not stochastic) and inventory is not None and repair_strategy in (
        "inventory_first",
        "feasible_topk",
        "inventory_weighted",
        "feasible_joint_topk",
    ):
        dc_probs = probs_grid.sum(dim=2)
        inv_int = inventory.int()
        demand_vec = demand.view(-1).to(inv_int).int()
        if repair_strategy == "feasible_joint_topk":
            topk = min(top_k, probs_flat.size(1))
            top_idx = torch.topk(probs_flat, k=topk, dim=1).indices
            chosen = []
            for b in range(B):
                picked = None
                for idx in top_idx[b]:
                    dc_idx = int(idx.item() // C)
                    if inv_int[b, dc_idx] >= demand_vec[b]:
                        picked = idx
                        break
                if picked is None:
                    picked = probs_flat[b].argmax()
                chosen.append(picked)
            chosen = torch.stack(chosen)
            primary_dc = chosen // C
            primary_carrier = chosen % C
        elif repair_strategy == "inventory_first":
            feasible = inv_int >= demand_vec.unsqueeze(1)
            any_feasible = feasible.any(dim=1)
            masked = torch.where(any_feasible.unsqueeze(1), dc_probs * feasible.float(), dc_probs)
            primary_dc = masked.argmax(dim=1)
        elif repair_strategy == "feasible_topk":
            chosen = []
            for b in range(B):
                sorted_idx = torch.argsort(dc_probs[b], descending=True)
                found = None
                for idx in sorted_idx[: min(top_k, D)]:
                    if inv_int[b, idx] >= demand_vec[b]:
                        found = idx
                        break
                chosen.append(found if found is not None else sorted_idx[0])
            primary_dc = torch.stack(chosen)
        elif repair_strategy == "inventory_weighted":
            weights = (inv_int.float() / (demand_vec.unsqueeze(1).float() + 1e-9)).clamp(max=1.0)
            if inventory_weight_power is not None and float(inventory_weight_power) != 1.0:
                weights = weights.pow(float(inventory_weight_power))
            weighted = dc_probs * weights
            fallback = weighted.sum(dim=1, keepdim=True) <= 1e-9
            weighted = torch.where(fallback, dc_probs, weighted)
            primary_dc = weighted.argmax(dim=1)

        if primary_carrier is None and primary_dc is not None:
            carriers = []
            for b in range(B):
                carriers.append(select_carrier_for_dc(b, int(primary_dc[b].item())))
            primary_carrier = torch.tensor(carriers, device=logits_sel.device, dtype=torch.long)

    if primary_dc is None:
        # Select primary (DC, Carrier) option
        if stochastic:
            top_probs, top_indices = torch.topk(probs_flat, k=min(top_k, D * C), dim=1)
            top_probs = top_probs / (top_probs.sum(dim=1, keepdim=True) + 1e-9)
            choice_indices = torch.multinomial(top_probs, num_samples=1).squeeze(1)
            primary_option = torch.gather(top_indices, 1, choice_indices.unsqueeze(1)).squeeze(1)
        else:
            primary_option = probs_flat.argmax(dim=1)

        primary_dc = primary_option // C
        primary_carrier = primary_option % C

    # Raw pre-repair plan: assign all demand to the selected primary DC.
    plan_before = torch.zeros(B, D, dtype=torch.int32, device=logits_sel.device)
    for b in range(B):
        demand_qty = int(demand[b].item())
        if demand_qty <= 0:
            continue
        dc_idx = int(primary_dc[b].item())
        plan_before[b, dc_idx] = demand_qty

    plan_after = torch.zeros(B, D, dtype=torch.int32, device=logits_sel.device)
    carrier_per_dc = torch.full((B, D), -1, dtype=torch.long, device=logits_sel.device)

    if inventory is None or not repair:
        plan_after = plan_before.clone()
        for b in range(B):
            dc_idx = int(primary_dc[b].item())
            carrier_idx = int(primary_carrier[b].item())
            demand_qty = int(demand[b].item())
            if demand_qty > 0:
                carrier_per_dc[b, dc_idx] = carrier_idx
        plan_before = plan_before.float()
        return plan_before, plan_after, carrier_per_dc

    inv_int = inventory.int()
    dc_probs = probs_grid.sum(dim=2)
    if elig_grid is not None:
        dc_eligible = (elig_grid.sum(dim=2) > 0).float()
        dc_probs = dc_probs * dc_eligible

    for b in range(B):
        demand_qty = int(demand[b].item())
        if demand_qty == 0:
            continue

        primary_dc_idx = int(primary_dc[b].item())
        available_primary = int(inv_int[b, primary_dc_idx].item())

        if available_primary >= demand_qty:
            plan_after[b, primary_dc_idx] = demand_qty
            carrier_per_dc[b, primary_dc_idx] = select_carrier_for_dc(b, primary_dc_idx)
        else:
            remaining = demand_qty
            if available_primary > 0:
                plan_after[b, primary_dc_idx] = available_primary
                carrier_per_dc[b, primary_dc_idx] = select_carrier_for_dc(b, primary_dc_idx)
                remaining -= available_primary

            probs_row = dc_probs[b].clone()
            probs_row[primary_dc_idx] = -1.0
            _, sorted_dc_idx = torch.sort(probs_row, descending=True)

            for dc_idx in sorted_dc_idx:
                if remaining <= 0:
                    break
                dc_id = int(dc_idx.item())
                available = int(inv_int[b, dc_id].item())
                if available > 0:
                    take = min(remaining, available)
                    plan_after[b, dc_id] = take
                    carrier_per_dc[b, dc_id] = select_carrier_for_dc(b, dc_id)
                    remaining -= take

            if debug and remaining > 0:
                print(f"[proxy][inference] Unfulfilled demand for batch {b}: {remaining} units")

    return plan_before.float(), plan_after, carrier_per_dc
