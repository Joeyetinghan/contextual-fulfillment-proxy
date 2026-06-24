import argparse
import json

import src.config as cfg


def parse_train_proxy_args():
    parser = argparse.ArgumentParser()
    # core
    parser.add_argument('--batch_size', type=int, default=cfg.PROXY_MODEL_BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=cfg.PROXY_MODEL_LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=cfg.PROXY_MODEL_WEIGHT_DECAY)
    # loss weights
    parser.add_argument('--selection_weight', type=float, default=cfg.PROXY_MODEL_SELECTION_WEIGHT)
    parser.add_argument('--carrier_loss_weight', type=float, default=1.0, help='Weight for carrier loss in hierarchical mode')
    parser.add_argument('--constraint_loss_weight', type=float, default=cfg.PROXY_MODEL_CONSTRAINT_VIOLATION_WEIGHT)
    parser.add_argument('--cardinality_penalty_weight', type=float, default=cfg.PROXY_MODEL_CARDINALITY_PENALTY_WEIGHT)
    parser.add_argument('--entropy_weight', type=float, default=cfg.PROXY_MODEL_ENTROPY_WEIGHT, help='Entropy regularization weight (encourages peaked distributions)')
    parser.add_argument('--cost_loss_weight', type=float, default=cfg.PROXY_MODEL_COST_LOSS_WEIGHT, help='Expected cost loss weight (aligns proxy with actual cost objective)')
    parser.add_argument('--threshold_on_sel', type=float, default=cfg.PROXY_MODEL_THRESHOLD)
    parser.add_argument(
        '--repair_strategy',
        type=str,
        default=getattr(cfg, "PROXY_MODEL_REPAIR_STRATEGY", "default"),
        choices=["argmax_then_split", "default", "inventory_first", "feasible_topk", "inventory_weighted", "feasible_joint_topk"],
        help="Inference selection strategy before repair",
    )
    parser.add_argument(
        '--disable_class_weights',
        dest='disable_class_weights',
        action='store_true',
        help='Disable inverse-frequency class weights for the selection loss.',
    )
    parser.add_argument(
        '--enable_class_weights',
        dest='disable_class_weights',
        action='store_false',
        help='Enable inverse-frequency class weights for the selection loss (default).',
    )
    parser.set_defaults(disable_class_weights=False)
    parser.add_argument('--class_weight_power', type=float, default=1.0, help='Power for inverse-frequency class weights (1.0 for standard inverse freq)')
    parser.add_argument('--class_weight_max', type=float, default=0.0, help='Max clamp for class weights (<=0 disables clamp)')
    parser.add_argument('--use_dc_class_weights', action='store_true', default=False, help='Use DC-level class weights for auxiliary DC loss')
    parser.add_argument('--dc_class_weight_power', type=float, default=1.0, help='Power for inverse-frequency DC class weights')
    parser.add_argument('--dc_class_weight_max', type=float, default=0.0, help='Max clamp for DC class weights (<=0 disables clamp)')
    parser.add_argument('--use_carrier_class_weights', action='store_true', default=False, help='Use carrier-level class weights for hierarchical carrier loss')
    parser.add_argument('--carrier_class_weight_power', type=float, default=1.0, help='Power for inverse-frequency carrier class weights')
    parser.add_argument('--carrier_class_weight_max', type=float, default=0.0, help='Max clamp for carrier class weights (<=0 disables clamp)')
    parser.add_argument('--gumbel_tau', type=float, default=cfg.PROXY_MODEL_GUMBEL_TAU)
    parser.add_argument('--label_smoothing', type=float, default=cfg.PROXY_MODEL_LABEL_SMOOTHING,
                        help='Label smoothing factor (0.0=no smoothing, 0.1=typical)')
    # architecture
    parser.add_argument('--hidden_dim', type=int, default=cfg.PROXY_MODEL_HIDDEN_DIM)
    parser.add_argument('--n_layers', type=int, default=cfg.PROXY_MODEL_N_LAYERS)
    parser.add_argument('--dropout_p', type=float, default=cfg.PROXY_MODEL_DROPOUT_P)
    parser.add_argument('--sku_emb_dim', type=int, default=cfg.PROXY_MODEL_SKU_EMBEDDING_DIM)
    parser.add_argument('--brand_emb_dim', type=int, default=cfg.PROXY_MODEL_BRAND_EMBEDDING_DIM)
    parser.add_argument('--dc_embedding_dim', type=int, default=32, help='DC embedding dim (hierarchical_proxy_v2)')
    parser.add_argument('--carrier_emb_dim', type=int, default=8, help='Carrier embedding dim (hierarchical_proxy_v2)')
    parser.add_argument('--option_proj_dim', type=int, default=8, help='Option feature projection dim (hierarchical_proxy_v2)')
    parser.add_argument('--use_option_features_in_carrier', dest='use_option_features_in_carrier', action='store_true', default=True,
                        help='Use option features in carrier head (hierarchical_proxy_v2)')
    parser.add_argument('--no-use_option_features_in_carrier', dest='use_option_features_in_carrier', action='store_false')
    parser.add_argument('--use_scenario_module', dest='use_scenario_module', action='store_true', default=True,
                        help='Enable scenario module (hierarchical_proxy_v2)')
    parser.add_argument('--no-use_scenario_module', dest='use_scenario_module', action='store_false')
    parser.add_argument('--use_dc_module', dest='use_dc_module', action='store_true', default=True,
                        help='Enable DC feature module (hierarchical_proxy_v2)')
    parser.add_argument('--no-use_dc_module', dest='use_dc_module', action='store_false')
    parser.add_argument('--use_dc_embedding', dest='use_dc_embedding', action='store_true', default=True,
                        help='Enable DC embedding in carrier head (hierarchical_proxy_v2)')
    parser.add_argument('--no-use_dc_embedding', dest='use_dc_embedding', action='store_false')
    parser.add_argument('--use_carrier_embedding', dest='use_carrier_embedding', action='store_true', default=True,
                        help='Enable carrier embedding in carrier head (hierarchical_proxy_v2)')
    parser.add_argument('--no-use_carrier_embedding', dest='use_carrier_embedding', action='store_false')
    parser.add_argument('--epochs', type=int, default=cfg.PROXY_MODEL_EPOCHS)
    parser.add_argument('--agg_type', default="mean")
    parser.add_argument(
        '--scenario_combine',
        default=getattr(cfg, "PROXY_MODEL_SCENARIO_COMBINE", "add"),
        choices=["add", "concat"],
        help="How to fuse demand and cost scenario branches in hierarchical_proxy_v2.",
    )
    parser.add_argument('--model_variant', default='hierarchical_proxy_v2',
                        choices=['hierarchical_proxy_v2', 'single_tower'])
    parser.add_argument('--use_num_proj', action='store_true', default=False,
                        help='Enable numeric projection block (single_tower only)')
    parser.add_argument('--use_cost_summary', dest='use_cost_summary', action='store_true', default=True,
                        help='Include per-DC cost summary embedding (hierarchical_proxy_v2 only)')
    parser.add_argument('--no-use_cost_summary', dest='use_cost_summary', action='store_false')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config for experiment')
    parser.add_argument('--debug_nan', action='store_true', default=False, help='Fail fast on NaN/Inf in batches')
    parser.add_argument('--debug_loss_stats', action='store_true', default=False, help='Print loss component stats for first batch')
    parser.add_argument('--debug_eligibility', action='store_true', default=False, help='Print dataset eligibility/target mismatch stats')
    parser.add_argument('--debug_batch_eligibility', action='store_true', default=False, help='Print one-batch eligibility/target stats')
    parser.add_argument('--use_eligibility_mask', dest='use_eligibility_mask', action='store_true', default=True,
                        help='Apply eligibility mask inside loss (set false for unmasked ablations)')
    parser.add_argument('--no-use_eligibility_mask', dest='use_eligibility_mask', action='store_false')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader workers (0 disables multiprocessing).')
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true', default=True,
                        help='Enable pinned host memory for faster H2D transfer.')
    parser.add_argument('--no-pin_memory', dest='pin_memory', action='store_false')
    parser.add_argument('--persistent_workers', dest='persistent_workers', action='store_true', default=False,
                        help='Keep DataLoader workers alive across epochs (num_workers>0 only).')
    parser.add_argument('--no-persistent_workers', dest='persistent_workers', action='store_false')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Batches prefetched per worker (num_workers>0 only).')
    parser.add_argument('--tf32', dest='tf32', action='store_true', default=True,
                        help='Enable TF32 matmul/cuDNN kernels on Ampere+ GPUs.')
    parser.add_argument('--no-tf32', dest='tf32', action='store_false')
    parser.add_argument('--aux_dc_weight', type=float, default=0.0, help='Auxiliary DC classification loss weight')
    parser.add_argument('--aux_carrier_weight', type=float, default=0.0, help='Auxiliary carrier classification loss weight (conditioned on true DC)')
    parser.add_argument('--split_ratio', type=float, default=cfg.PROXY_MODEL_VALIDATION_SPLIT_RATIO)
    parser.add_argument(
        '--refit_full_data',
        action='store_true',
        default=False,
        help=(
            'Refit mode: train on the full dataset without a held-out validation split. '
            'Intended for final model retraining after hyperparameter selection.'
        ),
    )
    # feature normalization
    parser.add_argument('--normalize_global_features', dest='normalize_global_features', action='store_true', default=True,
                        help='Standardize global numeric features')
    parser.add_argument('--no-normalize_global_features', dest='normalize_global_features', action='store_false')
    parser.add_argument('--normalize_dc_features', dest='normalize_dc_features', action='store_true', default=True,
                        help='Standardize DC numeric features')
    parser.add_argument('--no-normalize_dc_features', dest='normalize_dc_features', action='store_false')
    parser.add_argument('--normalize_option_features', dest='normalize_option_features', action='store_true', default=True,
                        help='Standardize option numeric features')
    parser.add_argument('--no-normalize_option_features', dest='normalize_option_features', action='store_false')
    # viz
    parser.add_argument('--model_name', default='proxy_model_v1')
    parser.add_argument('--viz_samples', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=10)
    # early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=cfg.PROXY_MODEL_EARLY_STOPPING_PATIENCE)

    # CSAA-aligned UB diagnostics (validation only; uses on-disk CSAA dumps)
    parser.add_argument(
        '--csaa_solutions_root',
        type=str,
        default=str(cfg.DATA_DIR / 'peak' / 'csaa_solutions' / 'proxy_train'),
        help="Root directory containing CSAA per-order dumps: {date}/{order_id}/...",
    )
    parser.add_argument(
        '--ub_eval_orders',
        type=int,
        default=50,
        help="How many validation orders to evaluate proxy UB each print epoch (0 disables).",
    )
    parser.add_argument(
        '--ub_eval_scenarios',
        type=int,
        default=int(cfg.SAA_N2),
        help="How many CSAA evaluation scenarios (N2) to use per order for UB mean/CI (default: SAA_N2).",
    )
    parser.add_argument(
        '--ub_eval_every',
        type=int,
        default=1,
        help='Compute UB diagnostics every N print windows (1 = every print window).',
    )
    parser.add_argument(
        '--ub_eval_batch_size',
        type=int,
        default=256,
        help='Forward batch size for UB diagnostics to cap GPU memory usage.',
    )
    parser.add_argument('--final_eval_on_best', dest='final_eval_on_best', action='store_true', default=True,
                        help='Run one final validation pass on the best checkpoint before exit.')
    parser.add_argument('--no-final_eval_on_best', dest='final_eval_on_best', action='store_false')
    parser.add_argument('--final_ub_eval_orders', type=int, default=0,
                        help='Final UB eval orders on best checkpoint (>0 enables; 0 disables).')
    parser.add_argument('--final_ub_eval_scenarios', type=int, default=0,
                        help='Final UB eval N2 scenarios on best checkpoint (>0 overrides --ub_eval_scenarios).')
    parser.add_argument('--final_ub_eval_batch_size', type=int, default=256,
                        help='Forward batch size for final UB diagnostics on best checkpoint.')

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            cfg_data = json.load(f)
        print("=" * 70)
        print(f"Loaded config: {args.config}")
        print(json.dumps(cfg_data, indent=2))
        print("=" * 70)
        for k, v in cfg_data.items():
            if hasattr(args, k):
                setattr(args, k, v)
    if args.split_ratio < 0.0 or args.split_ratio >= 1.0:
        raise ValueError(
            f"Invalid --split_ratio={args.split_ratio}. "
            "Use 0 for full-data refit mode, or a value in (0, 1) for train/validation split."
        )
    if args.split_ratio == 0.0 and not args.refit_full_data:
        print("[proxy-train] split_ratio=0 detected; enabling --refit_full_data.")
        args.refit_full_data = True
    return args
