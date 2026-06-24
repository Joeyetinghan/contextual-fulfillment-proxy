"""Inference repair utilities for the hierarchical proxy model."""

import torch


def hierarchical_proxy_inference(
    logits_sel,
    inventory,
    demand,
    threshold=0.5,
    repair=True,
    eligibility_mask=None,
    stochastic=False,
    top_k=5,
    debug=False,
    return_raw_carrier=False,
    repair_strategy="argmax_then_split",
    inventory_weight_power=1.0,
):
    """
    Decodes hierarchical model logits into a concrete fulfillment plan with greedy inventory splitting.
    
    Args:
        logits_sel: tuple (logits_dc [B,D], logits_carrier [B,D,C])
        inventory: tensor [B,D] - available inventory per DC
        demand: tensor [B,1] - order quantity
        threshold: float - minimum probability for allocation (deterministic mode)
        repair: bool - enable greedy splitting across DCs when primary DC lacks inventory
        eligibility_mask: [B,D,C] or [B,D*C] - binary mask for eligible (DC, Carrier) pairs
        stochastic: bool - enable top-k sampling instead of argmax
        top_k: int - number of top candidates to sample from in stochastic mode
        debug: bool - unused (kept for API compatibility)
        return_raw_carrier: bool - if True, also return raw carrier_per_dc as 4th output
        repair_strategy: str - selection strategy for primary DC
            (argmax_then_split/default, inventory_first, feasible_topk, inventory_weighted, feasible_joint_topk)
        inventory_weight_power: float - exponent on inventory weighting for inventory_weighted strategy
    
    Returns:
        (plan_before [B,D], plan_after [B,D], carrier_per_dc [B,D])
        plan_before is the raw (pre-repair) plan; plan_after is repaired if repair=True.
        If return_raw_carrier=True, also returns carrier_raw [B,D] as 4th output.
    """
    logits_dc, logits_carrier = logits_sel
    B, D = logits_dc.shape
    C = logits_carrier.shape[2]
    
    # Parse eligibility mask
    elig_grid = None
    if eligibility_mask is not None:
        elig_grid = eligibility_mask.view(B, D, C) if eligibility_mask.ndim == 2 else eligibility_mask
        dc_eligible = elig_grid.any(dim=2).float()  # [B,D] - DCs with ≥1 eligible carrier
    else:
        dc_eligible = torch.ones(B, D, device=logits_dc.device)
    
    # ─────────── DC Selection ───────────
    probs_dc = torch.softmax(logits_dc, dim=1)  # [B,D]
    probs_dc = probs_dc * dc_eligible  # Mask ineligible DCs
    
    # Renormalize
    sum_probs = probs_dc.sum(dim=1, keepdim=True)
    probs_dc = torch.where(sum_probs > 1e-9, probs_dc / (sum_probs + 1e-9), probs_dc)

    probs_carrier = torch.softmax(logits_carrier, dim=2)  # [B,D,C]
    primary_carrier = None
    
    if stochastic:
        # Top-k sampling
        top_probs, top_indices = torch.topk(probs_dc, k=min(top_k, D), dim=1)
        top_probs = top_probs / (top_probs.sum(dim=1, keepdim=True) + 1e-9)
        choice_indices = torch.multinomial(top_probs, num_samples=1).squeeze(1)
        primary_dc = torch.gather(top_indices, 1, choice_indices.unsqueeze(1)).squeeze(1)
        has_active = torch.ones(B, dtype=torch.bool, device=logits_dc.device)
    else:
        # Deterministic argmax with threshold
        active_mask = (probs_dc > threshold) & (dc_eligible > 0)
        masked_probs = probs_dc.masked_fill(~active_mask, float('-inf'))
        primary_dc = masked_probs.argmax(dim=1)  # [B]
        has_active = active_mask.any(dim=1)  # [B]

        if repair_strategy == "feasible_joint_topk" and inventory is not None:
            inv_int = inventory.int()
            demand_vec = demand.view(-1).to(inv_int).int()
            joint = probs_dc.unsqueeze(2) * probs_carrier
            if elig_grid is not None:
                joint = joint * elig_grid
            joint_flat = joint.view(B, -1)
            topk = min(top_k, joint_flat.size(1))
            top_idx = torch.topk(joint_flat, k=topk, dim=1).indices
            chosen_dc = []
            chosen_carrier = []
            for b in range(B):
                picked = None
                for idx in top_idx[b]:
                    dc_idx = int(idx.item() // C)
                    if dc_eligible[b, dc_idx] > 0 and inv_int[b, dc_idx] >= demand_vec[b]:
                        picked = idx
                        break
                if picked is None:
                    picked = joint_flat[b].argmax()
                chosen_dc.append(int(picked.item() // C))
                chosen_carrier.append(int(picked.item() % C))
            primary_dc = torch.tensor(chosen_dc, device=logits_dc.device, dtype=torch.long)
            primary_carrier = torch.tensor(chosen_carrier, device=logits_dc.device, dtype=torch.long)
            has_active = dc_eligible.any(dim=1)
        elif repair_strategy in ("inventory_first", "feasible_topk", "inventory_weighted") and inventory is not None:
            inv_int = inventory.int()
            demand_vec = demand.view(-1).to(inv_int).int()
            if repair_strategy == "inventory_first":
                feasible = (inv_int >= demand_vec.unsqueeze(1)) & (dc_eligible > 0)
                any_feasible = feasible.any(dim=1)
                use_mask = torch.where(any_feasible.unsqueeze(1), feasible, active_mask)
                masked_probs = probs_dc.masked_fill(~use_mask, float('-inf'))
                primary_dc = masked_probs.argmax(dim=1)
                has_active = use_mask.any(dim=1)
            elif repair_strategy == "feasible_topk":
                chosen = []
                for b in range(B):
                    sorted_idx = torch.argsort(probs_dc[b], descending=True)
                    found = None
                    for idx in sorted_idx[: min(top_k, D)]:
                        if inv_int[b, idx] >= demand_vec[b] and dc_eligible[b, idx] > 0:
                            found = idx
                            break
                    chosen.append(found if found is not None else primary_dc[b])
                primary_dc = torch.stack(chosen)
            elif repair_strategy == "inventory_weighted":
                weights = (inv_int.float() / (demand_vec.unsqueeze(1).float() + 1e-9)).clamp(max=1.0)
                if inventory_weight_power is not None and float(inventory_weight_power) != 1.0:
                    weights = weights.pow(float(inventory_weight_power))
                weighted = probs_dc * weights * dc_eligible
                fallback = weighted.sum(dim=1, keepdim=True) <= 1e-9
                weighted = torch.where(fallback, probs_dc, weighted)
                primary_dc = weighted.argmax(dim=1)
                has_active = dc_eligible.any(dim=1)
    
    # ─────────── Helper: Select Best Carrier for a DC ───────────
    def select_carrier_for_dc(batch_idx, dc_idx):
        """Select best carrier for a specific (batch, DC) pair."""
        probs_c = probs_carrier[batch_idx, dc_idx]  # [C]
        
        if elig_grid is not None:
            carrier_eligible = elig_grid[batch_idx, dc_idx]  # [C]
            masked_probs = probs_c * carrier_eligible
            sum_probs = masked_probs.sum()
            if sum_probs > 1e-9:
                masked_probs = masked_probs / sum_probs
        else:
            masked_probs = probs_c
        
        if stochastic:
            # Top-k sampling
            eligible_indices = (masked_probs > 1e-9).nonzero(as_tuple=True)[0]
            if len(eligible_indices) == 0:
                return 0  # Fallback
            k_actual = min(top_k, len(eligible_indices))
            top_probs_c = masked_probs[eligible_indices].topk(k_actual).values
            top_indices_c = masked_probs[eligible_indices].topk(k_actual).indices
            top_probs_c = top_probs_c / (top_probs_c.sum() + 1e-9)
            choice_idx = torch.multinomial(top_probs_c, num_samples=1).item()
            return eligible_indices[top_indices_c[choice_idx]].item()
        else:
            # Argmax
            return masked_probs.argmax().item()
    
    # ─────────── Repair: Greedy Inventory Splitting ───────────
    # Build raw (pre-repair) plan: allocate to primary DC only
    plan_raw = torch.zeros(B, D, dtype=torch.float32, device=logits_dc.device)
    qty_vec = demand.squeeze(1).float()
    carrier_raw = torch.full((B, D), -1, dtype=torch.long, device=logits_dc.device)
    for b in range(B):
        if has_active[b]:
            plan_raw[b, primary_dc[b]] = qty_vec[b]
            if primary_carrier is not None:
                carrier_raw[b, primary_dc[b]] = primary_carrier[b]
            else:
                carrier_raw[b, primary_dc[b]] = select_carrier_for_dc(b, primary_dc[b].item())
    
    if not repair:
        if return_raw_carrier:
            return plan_raw, plan_raw.int(), carrier_raw, carrier_raw
        return plan_raw, plan_raw.int(), carrier_raw
    
    # Greedy splitting: fulfill from primary DC first, then others by inventory
    inv_int = inventory.int()
    plan_after = torch.zeros(B, D, dtype=torch.int32, device=logits_dc.device)
    carrier_per_dc = torch.full((B, D), -1, dtype=torch.long, device=logits_dc.device)
    
    for b in range(B):
        if not has_active[b]:
            continue
        
        demand_qty = int(demand[b].item())
        if demand_qty == 0:
            continue
        
        primary_dc_idx = primary_dc[b].item()
        available_primary = int(inv_int[b, primary_dc_idx].item())
        
        # Try primary DC first
        if available_primary >= demand_qty:
            # Primary DC has enough
            plan_after[b, primary_dc_idx] = demand_qty
            carrier_per_dc[b, primary_dc_idx] = select_carrier_for_dc(b, primary_dc_idx)
        else:
            # Need to split across multiple DCs
            remaining = demand_qty
            
            # Allocate from primary DC
            if available_primary > 0:
                plan_after[b, primary_dc_idx] = available_primary
                carrier_per_dc[b, primary_dc_idx] = select_carrier_for_dc(b, primary_dc_idx)
                remaining -= available_primary
            
            # Sort other DCs by probability (descending)
            probs_row = probs_dc[b].clone()
            probs_row[primary_dc_idx] = -1.0  # Exclude primary (already used)
            sorted_probs, sorted_dc_idx = torch.sort(probs_row, descending=True)
            
            # Greedily allocate from DCs with highest probability
            for dc_idx in sorted_dc_idx:
                if remaining <= 0:
                    break
                
                dc_id = dc_idx.item()
                available = int(inv_int[b, dc_id].item())
                
                if available > 0:
                    take = min(remaining, available)
                    plan_after[b, dc_id] = take
                    carrier_per_dc[b, dc_id] = select_carrier_for_dc(b, dc_id)
                    remaining -= take
    
    plan_before = plan_raw  # Pre-repair plan
    if return_raw_carrier:
        return plan_before, plan_after, carrier_per_dc, carrier_raw
    return plan_before, plan_after, carrier_per_dc
