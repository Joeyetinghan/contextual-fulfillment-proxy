def _safe_ratio(num: float, den: float) -> float:
    den = float(den)
    if abs(den) < 1e-12:
        return float("nan")
    return float(num) / den


def _print_split_metrics(split_name: str, metrics: dict):
    print(f"  {split_name} Metrics (Raw):")
    print(
        f'    DC     - Hit@1: {metrics["hit1_dc"]:.3f} | '
        f'Hit@5: {metrics["hit5_dc"]:.3f} | MRR: {metrics["mrr_dc"]:.3f}'
    )
    print(
        f'    Carrier- Hit@1: {metrics["hit1_carrier"]:.3f} | '
        f'Hit@5: {metrics["hit5_carrier"]:.3f} | MRR: {metrics["mrr_carrier"]:.3f}'
    )
    print(
        f'    Joint  - Hit@1: {metrics["joint_hit1"]:.3f} | '
        f'Hit@5: {metrics["joint_hit5"]:.3f} | MRR: {metrics["mrr_joint"]:.3f}'
    )

    print(f"  {split_name} Metrics (Repaired):")
    print(f'    DC     - Hit@1: {metrics["hit1_dc_repaired"]:.3f}')
    print(f'    Carrier- Hit@1: {metrics["hit1_carrier_repaired"]:.3f}')
    print(f'    Joint  - Hit@1: {metrics["joint_hit1_repaired"]:.3f}')

    print(f"  {split_name} Repair Diagnostics:")
    print(
        f'    Changed: {metrics["repair_changed_rate"]:.3f} | '
        f'OverInv: {metrics["repair_over_inv_rate"]:.3f} '
        f'(qty% {metrics["repair_over_inv_qty_frac"]:.3f}) | '
        f'RawInelig: {metrics["repair_raw_ineligible_rate"]:.3f} | '
        f'RawPrimInelig: {metrics["repair_raw_primary_ineligible_rate"]:.3f}'
    )
    print(
        f'    PrimaryChanged: {metrics["repair_primary_changed_rate"]:.3f} | '
        f'DropPrimary: {metrics["repair_drop_primary_rate"]:.3f} | '
        f'CarrierChanged: {metrics["repair_carrier_changed_rate"]:.3f} | '
        f'Split: {metrics["repair_split_rate"]:.3f}'
    )
    print(
        f'    PrimaryOOS: {metrics["repair_raw_primary_oos_rate"]:.3f} | '
        f'Unfulfilled: {metrics["repair_unfulfilled_rate"]:.3f} '
        f'(qty% {metrics["repair_unfulfilled_qty_frac"]:.3f}) | '
        f'MovedQty% {metrics["repair_moved_qty_frac"]:.3f}'
    )
    print(
        f'    Retention (Repaired/Raw Hit@1): '
        f'DC {_safe_ratio(metrics["hit1_dc_repaired"], metrics["hit1_dc"]):.3f}x | '
        f'Carrier {_safe_ratio(metrics["hit1_carrier_repaired"], metrics["hit1_carrier"]):.3f}x | '
        f'Joint {_safe_ratio(metrics["joint_hit1_repaired"], metrics["joint_hit1"]):.3f}x'
    )


def print_train_val_metrics(train_metrics: dict, val_metrics: dict):
    _print_split_metrics("Train", train_metrics)
    _print_split_metrics("Val", val_metrics)
    print(
        f'  Val Delta (Repaired-Raw Hit@1): '
        f'DC {val_metrics["hit1_dc_repaired"] - val_metrics["hit1_dc"]:+.3f} | '
        f'Carrier {val_metrics["hit1_carrier_repaired"] - val_metrics["hit1_carrier"]:+.3f} | '
        f'Joint {val_metrics["joint_hit1_repaired"] - val_metrics["joint_hit1"]:+.3f}'
    )
    print(
        f'  Gen Gap (Val-Train Raw Hit@1): '
        f'DC {val_metrics["hit1_dc"] - train_metrics["hit1_dc"]:+.3f} | '
        f'Carrier {val_metrics["hit1_carrier"] - train_metrics["hit1_carrier"]:+.3f} | '
        f'Joint {val_metrics["joint_hit1"] - train_metrics["joint_hit1"]:+.3f}'
    )


def print_val_metrics(val_metrics: dict):
    """Print validation-only metrics using the canonical 'Val Metrics' layout."""
    _print_split_metrics("Val", val_metrics)


def print_best_so_far(best_loss: float, best_joint_hit1_repaired: float, best_epoch: int):
    print(
        f"  Best So Far: "
        f"va_loss={best_loss:.4f} | "
        f"val_joint_hit1_repaired={best_joint_hit1_repaired:.3f} "
        f"(E{best_epoch:02d})"
    )
