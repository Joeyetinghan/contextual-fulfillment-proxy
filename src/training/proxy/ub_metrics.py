import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.config as cfg
from src.model.hierarchical_proxy_inference import hierarchical_proxy_inference
from src.model.proxy_inference import proxy_inference
from src.saa_procedure import _sample_penalty_matrix, evaluate_candidate_solution


def compute_proxy_ub_metrics(
    model,
    tensors: dict,
    info: dict,
    metadata_rows: list[dict],
    val_indices: list[int],
    device: torch.device,
    repair_strategy: str,
    csaa_root: Path,
    max_orders: int,
    n2_eval: int,
    forward_batch_size: int = 256,
):
    """
    Compute an upper-bound estimate (mean + CI) for a proxy plan on a subset of validation orders,
    using CSAA evaluation scenarios (N2) stored on disk.
    """
    if max_orders <= 0 or n2_eval <= 0:
        return None
    if not metadata_rows or len(metadata_rows) < len(tensors.get("quantity_vector", [])):
        return None

    if not hasattr(compute_proxy_ub_metrics, "_order_total_counts"):
        counts = {}
        for row in metadata_rows:
            order_id = row.get("order_id")
            if order_id:
                counts[order_id] = counts.get(order_id, 0) + 1
        compute_proxy_ub_metrics._order_total_counts = counts
    total_counts = compute_proxy_ub_metrics._order_total_counts

    order_to_idxs = {}
    order_to_date = {}
    for idx in val_indices:
        if idx >= len(metadata_rows):
            continue
        row = metadata_rows[idx] or {}
        order_id = row.get("order_id")
        date = row.get("date")
        if not order_id or not date:
            continue
        order_to_idxs.setdefault(order_id, []).append(idx)
        if order_id not in order_to_date:
            order_to_date[order_id] = date
        elif order_to_date[order_id] != date:
            order_to_date[order_id] = ""

    candidate_orders = [
        order_id
        for order_id, idxs in order_to_idxs.items()
        if order_to_date.get(order_id) and total_counts.get(order_id) == len(idxs)
    ]
    if not candidate_orders:
        return None

    rng = random.Random(cfg.RANDOM_SEED)
    rng.shuffle(candidate_orders)
    selected_orders = candidate_orders[:max_orders]

    flat_indices = []
    order_offsets = {}
    for order_id in selected_orders:
        start = len(flat_indices)
        idxs = sorted(order_to_idxs[order_id])
        flat_indices.extend(idxs)
        order_offsets[order_id] = (start, len(flat_indices))

    batch = {
        k: tensors[k][flat_indices]
        for k in [
            "global_features",
            "dc_features",
            "option_features",
            "scenario_demand",
            "delivery_penalty",
            "sku_idx",
            "brand_idx",
            "eligibility_mask",
            "quantity_vector",
        ]
        if k in tensors
    }
    if len(batch.get("quantity_vector", [])) == 0:
        return None

    model_was_training = model.training
    model.eval()
    chunk_size = max(1, int(forward_batch_size))
    total_rows = len(flat_indices)
    plan_chunks = []
    carrier_chunks = []
    with torch.inference_mode():
        for s in range(0, total_rows, chunk_size):
            e = min(total_rows, s + chunk_size)
            output = model(
                global_feats=batch["global_features"][s:e].to(device),
                dc_feats=batch["dc_features"][s:e].to(device),
                option_feats=batch["option_features"][s:e].to(device),
                demand_scenarios=batch["scenario_demand"][s:e].to(device).float(),
                delivery_penalty=batch["delivery_penalty"][s:e].to(device).float(),
                sku_idx=batch["sku_idx"][s:e].to(device),
                brand_idx=batch["brand_idx"][s:e].to(device),
            )

            inventory = batch["dc_features"][s:e, :, 0].to(device).float()
            demand = batch["quantity_vector"][s:e].to(device).float()
            eligibility_mask = batch.get("eligibility_mask")
            if eligibility_mask is not None:
                eligibility_mask = eligibility_mask[s:e].to(device)

            if isinstance(output, tuple):
                _, plan_after, carrier_per_dc, _ = hierarchical_proxy_inference(
                    output,
                    inventory=inventory,
                    demand=demand,
                    repair=True,
                    eligibility_mask=eligibility_mask,
                    repair_strategy=repair_strategy,
                    return_raw_carrier=True,
                )
            else:
                _, plan_after, carrier_per_dc = proxy_inference(
                    output,
                    inventory=inventory,
                    demand=demand,
                    repair=True,
                    eligibility_mask=eligibility_mask,
                    num_dcs=info["num_dcs"],
                    num_carriers=info["num_carriers"],
                    debug=False,
                    repair_strategy=repair_strategy,
                )
            plan_chunks.append(plan_after.detach().cpu())
            carrier_chunks.append(carrier_per_dc.detach().cpu())

    if model_was_training:
        model.train()

    plan_after = torch.cat(plan_chunks, dim=0)
    carrier_per_dc = torch.cat(carrier_chunks, dim=0)

    dcs = info.get("dcs") or list(range(info["num_dcs"]))
    carriers = info.get("carriers") or list(range(1, info["num_carriers"] + 1))

    proxy_means = []
    proxy_ci95s = []
    csaa_means = []
    csaa_ci95s = []
    proxy_minus_csaa = []
    n2_used_list = []
    missing = 0
    skipped = 0

    for order_id in selected_orders:
        date = order_to_date.get(order_id)
        if not date:
            skipped += 1
            continue
        order_dir = csaa_root / date / order_id
        meta_path = order_dir / "metadata.pkl"
        scen_path = order_dir / "candidate_scenarios.npz"
        base_cost_path = order_dir / "base_costs.npy"
        penalty_path = order_dir / "delivery_penalties.npz"
        if not (meta_path.exists() and scen_path.exists() and base_cost_path.exists() and penalty_path.exists()):
            missing += 1
            continue

        try:
            with meta_path.open("rb") as f:
                meta = pickle.load(f)
            option_snapshot = meta.get("option_snapshot") or []
            saa_bounds = meta.get("saa_bounds") or {}
            saa_q = int(saa_bounds.get("saa_q", cfg.SAA_Q))
            saa_n1 = int(saa_bounds.get("saa_n1", cfg.SAA_N1))
            saa_n2 = int(saa_bounds.get("saa_n2", cfg.SAA_N2))
            pair_to_opt = {
                (int(o["dc_id"]), int(o["carrier_service_id"])): int(o["option_id"])
                for o in option_snapshot
            }

            start, end = order_offsets[order_id]
            z_candidate = {}
            order_items_rows = []
            inv_rows = []
            sku_list = []

            for pos in range(start, end):
                row_idx = flat_indices[pos]
                sku = str(metadata_rows[row_idx].get("sku_id"))
                qty = float(tensors["quantity_vector"][row_idx].item())
                sku_list.append(sku)
                order_items_rows.append((sku, int(round(qty))))
                inv_rows.append(tensors["dc_features"][row_idx, :, 0].numpy().astype(float))

                alloc = plan_after[pos]
                car = carrier_per_dc[pos]
                for d_idx in (alloc > 0).nonzero(as_tuple=True)[0].tolist():
                    q = int(alloc[d_idx].item())
                    if q <= 0:
                        continue
                    c_idx = int(car[d_idx].item())
                    if c_idx < 0 or c_idx >= len(carriers):
                        continue
                    dc_id = int(dcs[d_idx])
                    carrier_id = int(carriers[c_idx])
                    opt_id = pair_to_opt.get((dc_id, carrier_id))
                    if opt_id is None:
                        continue
                    key = (sku, int(opt_id))
                    z_candidate[key] = z_candidate.get(key, 0.0) + float(q)

            if not order_items_rows:
                skipped += 1
                continue

            with np.load(scen_path) as dem_npz:
                if not dem_npz.files:
                    skipped += 1
                    continue
                any_key = dem_npz.files[0]
                total_scenarios = int(dem_npz[any_key].shape[0])
                num_candidate_scenarios = min(total_scenarios, int(saa_q * saa_n1))
                n2_total = max(0, total_scenarios - num_candidate_scenarios)
                if saa_n2 > 0:
                    n2_total = min(n2_total, int(saa_n2))
                n2_use = min(int(n2_eval), int(n2_total))
                if n2_use <= 0:
                    skipped += 1
                    continue
                n2_used_list.append(int(n2_use))
                eval_slice = slice(num_candidate_scenarios, num_candidate_scenarios + n2_use)
                scen_index = [f"scenario_{i}" for i in range(n2_use)]

                demand_scenarios_N2 = {}
                for sku in sku_list:
                    key = f"demand_{sku}"
                    arr = dem_npz[key] if key in dem_npz.files else np.zeros(total_scenarios, dtype=np.int16)
                    demand_scenarios_N2[sku] = pd.Series(arr[eval_slice], index=scen_index)

            base_costs = np.load(base_cost_path).astype(np.float32, copy=False)
            with np.load(penalty_path) as pen:
                penalty = pen["penalty"].astype(np.float32, copy=False)
                if "option_id" in pen.files:
                    opt_ids = pen["option_id"].astype(np.int32, copy=False)
                else:
                    opt_ids = np.asarray([o["option_id"] for o in option_snapshot], dtype=np.int32)

            ship_eval = base_costs.reshape(1, -1) + penalty[eval_slice]
            ship_df = pd.DataFrame(ship_eval, index=scen_index, columns=opt_ids.tolist())
            shipping_costs_N2 = {sku: ship_df for sku in sku_list}

            base_costs_by_option = pd.Series(base_costs, index=opt_ids.tolist())
            option_to_dc = {int(o["option_id"]): int(o["dc_id"]) for o in option_snapshot}
            option_to_carrier = {int(o["option_id"]): int(o["carrier_service_id"]) for o in option_snapshot}
            dc_to_option_ids = {}
            for opt_id, dc_id in option_to_dc.items():
                dc_to_option_ids.setdefault(int(dc_id), []).append(int(opt_id))

            inv_mat = np.stack(inv_rows, axis=0)
            inv_df = pd.DataFrame(inv_mat, index=sku_list, columns=[int(x) for x in dcs])
            order_items_df = pd.DataFrame(order_items_rows, columns=["sku_ID", "quantity"])

            promise_days = int(meta.get("promise_delivery_days", 2))
            use_future_penalties = bool(
                saa_bounds.get("use_future_penalties", getattr(cfg, "SAA_USE_FUTURE_PENALTIES", False))
            )
            future_penalties_N2 = None
            if use_future_penalties:
                raw_seed = saa_bounds.get("future_penalty_seed", None)
                if raw_seed is None:
                    raw_seed = meta.get("saa_master_seed", cfg.RANDOM_SEED) + 1
                penalty_seed = int(raw_seed)
                pen_rng = np.random.default_rng(penalty_seed)
                penalties_all = _sample_penalty_matrix(
                    option_ids=opt_ids.tolist(),
                    option_to_carrier=option_to_carrier,
                    order_set="proxy_train",
                    promise_days=promise_days,
                    num_scenarios=total_scenarios,
                    rng=pen_rng,
                )
                future_penalties_N2 = penalties_all[eval_slice]

            obj_mean, obj_std = evaluate_candidate_solution(
                z_candidate=z_candidate,
                on_hand_inventory=inv_df,
                demand_scenarios_N2=demand_scenarios_N2,
                shipping_costs_N2=shipping_costs_N2,
                stockout_penalty=cfg.STOCKOUT_PENALTY_PER_UNIT,
                base_costs_by_option=base_costs_by_option,
                dc_to_option_ids=dc_to_option_ids,
                option_to_dc=option_to_dc,
                option_to_carrier=option_to_carrier,
                order_set="proxy_train",
                promise_days=promise_days,
                future_penalties=future_penalties_N2,
                order_items=order_items_df,
                return_stats=True,
            )
            ci95 = 1.96 * float(obj_std) / math.sqrt(float(n2_use))
            proxy_means.append(float(obj_mean))
            proxy_ci95s.append(float(ci95))

            if "bar_f" in saa_bounds and "candidate_eval_std" in saa_bounds:
                csaa_bar = float(saa_bounds.get("bar_f", float("nan")))
                best_pos = saa_bounds.get("best_pos")
                stds = saa_bounds.get("candidate_eval_std") or []
                csaa_std = float(stds[best_pos]) if best_pos is not None and best_pos < len(stds) else float("nan")
                denom = float(saa_n2) if saa_n2 > 0 else float(cfg.SAA_N2)
                csaa_ci95 = 1.96 * csaa_std / math.sqrt(denom) if csaa_std == csaa_std else float("nan")
                csaa_means.append(csaa_bar)
                csaa_ci95s.append(csaa_ci95)
                if csaa_bar == csaa_bar:
                    proxy_minus_csaa.append(float(obj_mean) - csaa_bar)
        except Exception:
            skipped += 1
            continue

    if not proxy_means:
        return None

    out = {
        "orders_requested": int(max_orders),
        "orders_candidate": int(len(candidate_orders)),
        "orders_selected": int(len(selected_orders)),
        "orders_evaluated": int(len(proxy_means)),
        "orders_missing_artifacts": int(missing),
        "orders_skipped": int(skipped),
        "n2_eval_requested": int(n2_eval),
        "n2_used_mean": float(np.mean(n2_used_list)) if n2_used_list else float("nan"),
        "n2_used_min": int(np.min(n2_used_list)) if n2_used_list else 0,
        "n2_used_max": int(np.max(n2_used_list)) if n2_used_list else 0,
        "proxy_ub_mean": float(np.mean(proxy_means)),
        "proxy_ub_ci95_mean": float(np.mean(proxy_ci95s)) if proxy_ci95s else float("nan"),
    }
    if csaa_means:
        arr = np.asarray(proxy_minus_csaa, dtype=np.float32) if proxy_minus_csaa else np.asarray([])
        proxy_ci_arr = np.asarray(proxy_ci95s, dtype=np.float32) if proxy_ci95s else np.asarray([])
        ci95_arr = np.asarray(csaa_ci95s, dtype=np.float32) if csaa_ci95s else np.asarray([])
        z_scores = np.asarray([])
        sig_better_frac = float("nan")
        sig_worse_frac = float("nan")
        sig_tie_frac = float("nan")
        within_diff_ci95_frac = float("nan")
        if arr.size and ci95_arr.size:
            denom = ci95_arr / 1.96
            mask = np.isfinite(arr) & np.isfinite(denom) & (denom > 0)
            if mask.any():
                z_scores = arr[mask] / denom[mask]
        if arr.size and proxy_ci_arr.size and ci95_arr.size:
            mask_ci = np.isfinite(arr) & np.isfinite(proxy_ci_arr) & np.isfinite(ci95_arr)
            if mask_ci.any():
                diff_ci95 = np.sqrt(np.square(proxy_ci_arr[mask_ci]) + np.square(ci95_arr[mask_ci]))
                robust = diff_ci95 > 0
                if robust.any():
                    d = arr[mask_ci][robust]
                    dc = diff_ci95[robust]
                    sig_better_frac = float(np.mean(d < -dc))
                    sig_worse_frac = float(np.mean(d > dc))
                    sig_tie_frac = float(1.0 - sig_better_frac - sig_worse_frac)
                    within_diff_ci95_frac = float(np.mean(np.abs(d) <= dc))

        csaa_mean = float(np.mean(csaa_means))
        proxy_minus_csaa_mean = float(np.mean(proxy_minus_csaa)) if proxy_minus_csaa else float("nan")
        out.update({
            "csaa_bar_mean": csaa_mean,
            "csaa_bar_ci95_mean": float(np.nanmean(csaa_ci95s)) if csaa_ci95s else float("nan"),
            "proxy_minus_csaa_bar": proxy_minus_csaa_mean,
            "proxy_minus_csaa_bar_pct": float((proxy_minus_csaa_mean / csaa_mean) * 100.0) if csaa_mean != 0 else float("nan"),
            "proxy_minus_csaa_median": float(np.median(arr)) if arr.size else float("nan"),
            "proxy_minus_csaa_p10": float(np.quantile(arr, 0.10)) if arr.size else float("nan"),
            "proxy_minus_csaa_p90": float(np.quantile(arr, 0.90)) if arr.size else float("nan"),
            "proxy_le_csaa_frac": float(np.mean(arr <= 0)) if arr.size else float("nan"),
            "proxy_minus_csaa_z_mean": float(np.mean(z_scores)) if z_scores.size else float("nan"),
            "proxy_minus_csaa_z_median": float(np.median(z_scores)) if z_scores.size else float("nan"),
            "proxy_minus_csaa_z_count": int(z_scores.size),
            "proxy_csaa_sig_better_frac": sig_better_frac,
            "proxy_csaa_sig_worse_frac": sig_worse_frac,
            "proxy_csaa_sig_tie_frac": sig_tie_frac,
            "proxy_csaa_within_diff_ci95_frac": within_diff_ci95_frac,
        })
    return out


def print_ub_metrics(ub_metrics: dict, ub_every: int):
    if not ub_metrics:
        print(f"  Val UB Metrics: skipped this epoch (ub_eval_every={ub_every})")
        return
    print(f'  Val UB Metrics (CSAA eval scenarios; {ub_metrics["orders_evaluated"]} orders):')
    print(
        f'    Coverage: requested={ub_metrics["orders_requested"]} | '
        f'candidate={ub_metrics["orders_candidate"]} | selected={ub_metrics["orders_selected"]} | '
        f'evaluated={ub_metrics["orders_evaluated"]}'
    )
    print(
        f'    N2 used: req={ub_metrics["n2_eval_requested"]} | '
        f'mean={ub_metrics["n2_used_mean"]:.1f} '
        f'(min={ub_metrics["n2_used_min"]}, max={ub_metrics["n2_used_max"]})'
    )
    print(
        f'    Proxy UB - mean: {ub_metrics["proxy_ub_mean"]:.3f} '
        f'(avg CI95: {ub_metrics["proxy_ub_ci95_mean"]:.3f}) '
        f'[missing={ub_metrics["orders_missing_artifacts"]} skipped={ub_metrics["orders_skipped"]}]'
    )
    if "csaa_bar_mean" not in ub_metrics:
        return
    print(
        f'    CSAA bar_f - mean: {ub_metrics["csaa_bar_mean"]:.3f} '
        f'(avg CI95: {ub_metrics["csaa_bar_ci95_mean"]:.3f}); '
        f'Proxy - CSAA: {ub_metrics["proxy_minus_csaa_bar"]:.3f}'
    )
    print(
        f'    Proxy-CSAA mean %: {ub_metrics["proxy_minus_csaa_bar_pct"]:+.2f}% '
        f'| Within diff-CI95: {ub_metrics["proxy_csaa_within_diff_ci95_frac"]:.2%}'
    )
    print(
        f'    Proxy-CSAA median: {ub_metrics["proxy_minus_csaa_median"]:.3f} '
        f'(p10={ub_metrics["proxy_minus_csaa_p10"]:.3f}, p90={ub_metrics["proxy_minus_csaa_p90"]:.3f}); '
        f'P(Proxy<=CSAA): {ub_metrics["proxy_le_csaa_frac"]:.2%}'
    )
    if ub_metrics.get("proxy_minus_csaa_z_count", 0) > 0:
        print(
            f'    Proxy-CSAA z: mean {ub_metrics["proxy_minus_csaa_z_mean"]:.3f}, '
            f'median {ub_metrics["proxy_minus_csaa_z_median"]:.3f} '
            f'(n={ub_metrics["proxy_minus_csaa_z_count"]})'
        )
    print(
        f'    Proxy vs CSAA (diff-CI95): '
        f'better {ub_metrics["proxy_csaa_sig_better_frac"]:.2%} | '
        f'tie {ub_metrics["proxy_csaa_sig_tie_frac"]:.2%} | '
        f'worse {ub_metrics["proxy_csaa_sig_worse_frac"]:.2%}'
    )
