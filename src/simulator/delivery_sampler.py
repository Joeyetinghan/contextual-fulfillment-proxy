"""Real-time outcome sampler for delivery times."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import joblib

import src.config as cfg

logger = logging.getLogger(__name__)

class OutcomeSampler:
    """Sampler for realized delivery time outcomes using per-carrier-service simulators."""
    
    def __init__(self, simulator_type: str = 'dl', scenario_source: str = 'simulator', ml_model: str = 'dl'):
        """
        Initialize the outcome sampler.
        
        Args:
            simulator_type: 'dl' or 'catboost'
            scenario_source: 'simulator' or 'ml'
            ml_model: 'dl' or 'qrf' (when scenario_source='ml')
        """
        self.simulator_type = simulator_type
        self.scenario_source = scenario_source
        self.ml_model = ml_model
        
        # Cache for per-carrier-service models
        # Key: carrier_service_id, Value: dict with model, scaler, encoders, etc.
        self._cs_model_cache: Dict[int, Dict[str, Any]] = {}
        
        # Fallback global model cache (for backward compatibility)
        self._global_model_cache: Optional[Dict[str, Any]] = None
        self._missing_diag: Dict[int, str] = {}
    
    @staticmethod
    def _format_carrier_id(carrier_service_id: int) -> str:
        """Normalize carrier IDs (handles float/int inputs that should map to same files)."""
        try:
            value = float(carrier_service_id)
            if value.is_integer():
                return str(int(value))
        except (TypeError, ValueError):
            pass
        return str(carrier_service_id)
    
    def _record_missing_model(
        self,
        carrier_service_id: int,
        carrier_id_str: str,
        original_id_str: str,
        cs_models_dir: Path,
        attempted_paths: List[str],
    ) -> str:
        attempted = attempted_paths or ['<none>']
        available = [p.name for p in sorted(cs_models_dir.glob(f"*{carrier_id_str}*"))[:10]]
        if not available and carrier_id_str != original_id_str:
            available = [p.name for p in sorted(cs_models_dir.glob(f"*{original_id_str}*"))[:10]]
        if not available:
            available = ['<none>']
        msg = (
            f"No carrier-specific model located for carrier_service_id={carrier_service_id} "
            f"(normalized={carrier_id_str}). Attempted paths: {attempted}. "
            f"Sample files containing the id: {available}. Models dir: {cs_models_dir.resolve()}"
        )
        self._missing_diag[carrier_service_id] = msg
        logger.warning(msg)
        return msg
    
    @staticmethod
    def _encode_with_unknown(encoder, values: pd.Series) -> np.ndarray:
        """Encode categorical values, mapping unseen labels to the unknown token."""
        unk = cfg.DELIVERY_DL_UNKNOWN_TOKEN
        classes = encoder.classes_
        arr = values.astype(str).to_numpy()
        mask = np.isin(arr, classes)
        if unk in classes:
            fallback = unk
        else:
            fallback = classes[0] if len(classes) else unk
        if not np.all(mask) and fallback != unk:
            logger.warning(
                "Label encoder missing unknown token '%s'; mapping %d unseen labels to '%s'.",
                unk, int((~mask).sum()), fallback
            )
        arr = np.where(mask, arr, fallback)
        return encoder.transform(arr)

    @staticmethod
    def _bundle_param(bundle: dict, key: str, default=None):
        if key in bundle:
            return bundle.get(key)
        model_params = bundle.get("model_params") or {}
        return model_params.get(key, default)

    @staticmethod
    def _infer_numerical_dim(bundle: dict) -> int:
        val = OutcomeSampler._bundle_param(bundle, "numerical_dim")
        if val is not None:
            return int(val)
        x_scaler = bundle.get("x_scaler")
        if x_scaler is not None:
            if hasattr(x_scaler, "n_features_in_"):
                return int(x_scaler.n_features_in_)
            if hasattr(x_scaler, "mean_"):
                return int(len(x_scaler.mean_))
        return int(len(cfg.DELIVERY_DL_NUMERICAL_FEATURES))
    
    def _load_cs_simulator(self, carrier_service_id: int) -> Optional[Dict[str, Any]]:
        """
        Load per-carrier-service simulator for a given carrier.
        
        Args:
            carrier_service_id: The carrier service ID
            
        Returns:
            Dict with model info, or None if not found
        """
        # Check cache first
        if carrier_service_id in self._cs_model_cache:
            return self._cs_model_cache[carrier_service_id]
        
        # Select directory based on scenario source
        if self.scenario_source == 'simulator':
            cs_models_dir = Path(cfg.DELIVERY_TIME_SIMULATOR_PATH)
        else:
            cs_models_dir = Path(cfg.DELIVERY_MODELS_CS_DIR)

        carrier_id_str = self._format_carrier_id(carrier_service_id)
        original_id_str = str(carrier_service_id)
        attempted_paths: List[str] = []
        
        if self.scenario_source == 'simulator':
            # Load simulator (DL or CatBoost)
            if self.simulator_type == 'dl':
                # Try DL simulator
                sim_meta_path = cs_models_dir / f"simulator_dl_{carrier_id_str}.joblib"
                attempted_paths.append(str(sim_meta_path))
                if sim_meta_path.exists():
                    sim_meta = joblib.load(sim_meta_path)
                    bundle_path_str = sim_meta.get('dl_model_path', '')
                    if bundle_path_str:
                        raw_bundle_path = Path(bundle_path_str)
                        bundle_candidates = [raw_bundle_path]
                        if not raw_bundle_path.is_absolute():
                            bundle_candidates.append(cs_models_dir / raw_bundle_path)
                        bundle_path = None
                        for candidate in bundle_candidates:
                            attempted_paths.append(str(candidate))
                            if candidate.exists():
                                bundle_path = candidate
                                break
                        if bundle_path is None:
                            # Fall back to the last candidate even if missing, so diagnostics show it
                            bundle_path = bundle_candidates[-1]
                        if not bundle_path.exists() and original_id_str != carrier_id_str:
                            alt_name = bundle_path.name.replace(original_id_str, carrier_id_str)
                            alt_path = bundle_path.with_name(alt_name)
                            if alt_path.exists():
                                bundle_path = alt_path
                        if bundle_path.exists():
                            import torch
                            from src.model.time_quantile_model import TimeQuantileModel
                            
                            bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
                            numerical_dim = self._infer_numerical_dim(bundle)
                            hidden_dim = int(self._bundle_param(bundle, "hidden_dim", 128))
                            n_layers = int(self._bundle_param(bundle, "n_layers", 3))
                            dropout_p = float(self._bundle_param(bundle, "dropout_p", cfg.DELIVERY_DL_DROPOUT_P))
                            dc_ori_emb = int(self._bundle_param(bundle, "dc_ori_embedding_dim", 8))
                            dc_des_emb = int(self._bundle_param(bundle, "dc_des_embedding_dim", 8))
                            model = TimeQuantileModel(
                                numerical_dim=numerical_dim,
                                hidden_dim=hidden_dim,
                                n_layers=n_layers,
                                dropout=True,
                                dropout_p=dropout_p,
                                dc_ori_vocab_size=bundle['vocab_sizes'].get('dc_ori'),
                                dc_des_vocab_size=bundle['vocab_sizes'].get('dc_des'),
                                dc_ori_embedding_dim=dc_ori_emb,
                                dc_des_embedding_dim=dc_des_emb,
                            )
                            model.load_state_dict(bundle['state_dict'])
                            model.eval()
                            
                            model_info = {
                                'type': 'dl',
                                'model': model,
                                'x_scaler': bundle.get('x_scaler'),
                                'encoders': bundle.get('categorical_encoders'),
                                'carrier_service_id': carrier_service_id,
                            }
                            self._cs_model_cache[carrier_service_id] = model_info
                            self._missing_diag.pop(carrier_service_id, None)
                            logger.info(
                                "[OutcomeSampler] Loaded DL simulator for carrier_service_id=%s from %s",
                                carrier_service_id,
                                bundle_path,
                            )
                            return model_info
            else:
                # Try CatBoost simulator
                sim_path = cs_models_dir / f"simulator_catboost_{carrier_id_str}.joblib"
                attempted_paths.append(str(sim_path))
                if sim_path.exists():
                    sim_bundle = joblib.load(sim_path)
                    model_info = {
                        'type': 'catboost',
                        'model': sim_bundle['model'],
                        'label_encoder': sim_bundle['label_encoder'],
                        'carrier_service_id': carrier_service_id,
                    }
                    self._cs_model_cache[carrier_service_id] = model_info
                    self._missing_diag.pop(carrier_service_id, None)
                    logger.info(
                        "[OutcomeSampler] Loaded CatBoost simulator for carrier_service_id=%s from %s",
                        carrier_service_id,
                        sim_path,
                    )
                    return model_info
        else:
            # ML models (for policy use, not outcome sampling typically)
            # But we support it for consistency
            if self.ml_model == 'dl':
                # Try per-cs DL model
                dl_model_path = cs_models_dir / f"dl_model_{carrier_id_str}.pt"
                attempted_paths.append(str(dl_model_path))
                if dl_model_path.exists():
                    import torch
                    from src.model.time_quantile_model import TimeQuantileModel
                    
                    bundle = torch.load(dl_model_path, map_location='cpu', weights_only=False)
                    numerical_dim = self._infer_numerical_dim(bundle)
                    hidden_dim = int(self._bundle_param(bundle, "hidden_dim", 128))
                    n_layers = int(self._bundle_param(bundle, "n_layers", 3))
                    dropout_p = float(self._bundle_param(bundle, "dropout_p", cfg.DELIVERY_DL_DROPOUT_P))
                    dc_ori_emb = int(self._bundle_param(bundle, "dc_ori_embedding_dim", 8))
                    dc_des_emb = int(self._bundle_param(bundle, "dc_des_embedding_dim", 8))
                    model = TimeQuantileModel(
                        numerical_dim=numerical_dim,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers,
                        dropout=True,
                        dropout_p=dropout_p,
                        dc_ori_vocab_size=bundle['vocab_sizes'].get('dc_ori'),
                        dc_des_vocab_size=bundle['vocab_sizes'].get('dc_des'),
                        dc_ori_embedding_dim=dc_ori_emb,
                        dc_des_embedding_dim=dc_des_emb,
                    )
                    # Support both 'model_state_dict' (new) and 'state_dict' (old) keys
                    state_dict_key = 'model_state_dict' if 'model_state_dict' in bundle else 'state_dict'
                    model.load_state_dict(bundle[state_dict_key])
                    model.eval()
                    
                    model_info = {
                        'type': 'dl',
                        'model': model,
                        'x_scaler': bundle.get('x_scaler'),
                        'encoders': bundle.get('categorical_encoders'),
                        'carrier_service_id': carrier_service_id,
                    }
                    self._cs_model_cache[carrier_service_id] = model_info
                    self._missing_diag.pop(carrier_service_id, None)
                    return model_info
            else:
                # Try per-cs QRF model
                qrf_path = cs_models_dir / f"rf_model_{carrier_id_str}.joblib"
                attempted_paths.append(str(qrf_path))
                if qrf_path.exists():
                    model = joblib.load(qrf_path)
                    model_info = {
                        'type': 'qrf',
                        'model': model,
                        'carrier_service_id': carrier_service_id,
                    }
                    self._cs_model_cache[carrier_service_id] = model_info
                    self._missing_diag.pop(carrier_service_id, None)
                    return model_info

        # Not found - return None (will fallback to global model)
        self._record_missing_model(
            carrier_service_id, carrier_id_str, original_id_str, cs_models_dir, attempted_paths
        )
        return None
    
    def _load_global_model(self) -> Optional[Dict[str, Any]]:
        """Load global model as fallback."""
        if self._global_model_cache is not None:
            return self._global_model_cache
        
        if self.scenario_source == 'simulator':
            # Try global simulator
            model_filename = (
                'delivery_simulator_dl.joblib' if self.simulator_type == 'dl'
                else 'delivery_simulator_catboost.joblib'
            )
            model_path = Path(cfg.DELIVERY_MODELS_DIR) / model_filename
            if model_path.exists():
                sim_bundle = joblib.load(model_path)
                
                if sim_bundle.get('type', 'catboost') == 'dl':
                    import torch
                    from src.model.time_quantile_model import TimeQuantileModel
                    
                    bundle_path_str = sim_bundle.get('dl_model_path', '')
                    if bundle_path_str:
                        bundle_path = Path(bundle_path_str)
                        # Handle relative paths (relative to models directory)
                        if not bundle_path.is_absolute():
                            bundle_path = Path(cfg.DELIVERY_MODELS_DIR) / bundle_path
                        if bundle_path.exists():
                            bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
                            numerical_dim = self._infer_numerical_dim(bundle)
                            hidden_dim = int(self._bundle_param(bundle, "hidden_dim", 128))
                            n_layers = int(self._bundle_param(bundle, "n_layers", 3))
                            dropout_p = float(self._bundle_param(bundle, "dropout_p", cfg.DELIVERY_DL_DROPOUT_P))
                            dc_ori_emb = int(self._bundle_param(bundle, "dc_ori_embedding_dim", 8))
                            dc_des_emb = int(self._bundle_param(bundle, "dc_des_embedding_dim", 8))
                            model = TimeQuantileModel(
                                numerical_dim=numerical_dim,
                                hidden_dim=hidden_dim,
                                n_layers=n_layers,
                                dropout=True,
                                dropout_p=dropout_p,
                                dc_ori_vocab_size=bundle['vocab_sizes'].get('dc_ori'),
                                dc_des_vocab_size=bundle['vocab_sizes'].get('dc_des'),
                                dc_ori_embedding_dim=dc_ori_emb,
                                dc_des_embedding_dim=dc_des_emb,
                            )
                            model.load_state_dict(bundle['state_dict'])
                            model.eval()
                            
                            self._global_model_cache = {
                                'type': 'dl',
                                'model': model,
                                'x_scaler': bundle.get('x_scaler'),
                                'encoders': bundle.get('categorical_encoders'),
                            }
                            return self._global_model_cache
                else:
                    # CatBoost
                    self._global_model_cache = {
                        'type': 'catboost',
                        'model': sim_bundle['model'],
                        'label_encoder': sim_bundle['label_encoder'],
                    }
                    return self._global_model_cache
        else:
            # ML models
            if self.ml_model == 'dl':
                dl_model_path = Path(cfg.DELIVERY_MODELS_DIR) / 'delivery_model_global_with_proxy.pt'
                if dl_model_path.exists():
                    import torch
                    from src.model.time_quantile_model import TimeQuantileModel
                    
                    bundle = torch.load(dl_model_path, map_location='cpu', weights_only=False)
                    numerical_dim = self._infer_numerical_dim(bundle)
                    hidden_dim = int(self._bundle_param(bundle, "hidden_dim", 128))
                    n_layers = int(self._bundle_param(bundle, "n_layers", 3))
                    dropout_p = float(self._bundle_param(bundle, "dropout_p", cfg.DELIVERY_DL_DROPOUT_P))
                    dc_ori_emb = int(self._bundle_param(bundle, "dc_ori_embedding_dim", 8))
                    dc_des_emb = int(self._bundle_param(bundle, "dc_des_embedding_dim", 8))
                    model = TimeQuantileModel(
                        numerical_dim=numerical_dim,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers,
                        dropout=True,
                        dropout_p=dropout_p,
                        dc_ori_vocab_size=bundle['vocab_sizes'].get('dc_ori'),
                        dc_des_vocab_size=bundle['vocab_sizes'].get('dc_des'),
                        dc_ori_embedding_dim=dc_ori_emb,
                        dc_des_embedding_dim=dc_des_emb,
                    )
                    model.load_state_dict(bundle['state_dict'])
                    model.eval()
                    
                    self._global_model_cache = {
                        'type': 'dl',
                        'model': model,
                        'x_scaler': bundle.get('x_scaler'),
                        'encoders': bundle.get('categorical_encoders'),
                    }
                    return self._global_model_cache
            else:
                # QRF
                qrf_path = Path(cfg.DELIVERY_MODELS_DIR) / 'delivery_model_global.joblib'
                if qrf_path.exists():
                    model = joblib.load(qrf_path)
                    self._global_model_cache = {
                        'type': 'qrf',
                        'model': model,
                    }
                    return self._global_model_cache
        
        return None
    
    def sample(self, features_df: pd.DataFrame, num_replications: int, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """
        Sample delivery times for options using per-carrier-service simulators.
        
        Args:
            features_df: DataFrame with features, must include carrier_service_id, dc_ori, dc_des, and all DELIVERY_TIME_FEATURES
            num_replications: Number of samples per option
            rng: Random number generator
            
        Returns:
            DataFrame with shape (n_options, num_replications) containing sampled delivery times
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Ensure all required features are present
        X = features_df.copy()
        for f in cfg.DELIVERY_TIME_FEATURES:
            if f not in X.columns:
                X[f] = 0
        
        # Check if carrier_service_id is present
        if 'carrier_service_id' not in X.columns:
            raise ValueError("features_df must include 'carrier_service_id' column")
        
        # Group by carrier_service_id and sample using appropriate model
        all_samples = []
        
        for carrier_id, group in X.groupby('carrier_service_id'):
            group_X = group.copy()
            
            # Try to load per-cs model
            model_info = self._load_cs_simulator(carrier_id)
            
            # Fallback to global model if per-cs not found
            if model_info is None:
                model_info = self._load_global_model()
                if model_info is None:
                    detail = self._missing_diag.get(carrier_id)
                    detail_msg = f" Details: {detail}" if detail else ""
                    raise ValueError(
                        f"No model found for carrier_service_id={carrier_id} and no global fallback available." + detail_msg
                    )
            
            # Sample using appropriate model
            if model_info['type'] == 'dl':
                group_samples = self._sample_dl(
                    group_X, num_replications, rng, 
                    model_info['model'], model_info['x_scaler'], model_info['encoders']
                )
            elif model_info['type'] == 'catboost':
                group_samples = self._sample_catboost(
                    group_X, num_replications, rng,
                    model_info['model'], model_info['label_encoder']
                )
            elif model_info['type'] == 'qrf':
                group_samples = self._sample_qrf(
                    group_X, num_replications, rng,
                    model_info['model']
                )
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")
            
            all_samples.append(group_samples)
        
        # Concatenate results in original order
        result = pd.concat(all_samples)
        # Reindex to match original order
        result = result.reindex(X.index)
        # Normalize to MultiIndex for consistent downstream lookups when index is tuple-like
        try:
            if len(result.index) > 0 and isinstance(result.index[0], tuple) and len(result.index[0]) == 2:
                result.index = pd.MultiIndex.from_tuples(
                    list(result.index), names=['dc_id', 'carrier_service_id']
                )
        except Exception:
            # If index isn't tuple-like, keep it as-is (e.g., evaluation datasets)
            pass
        
        return result
    
    def _sample_dl(
        self, 
        X: pd.DataFrame, 
        num_replications: int, 
        rng: np.random.Generator,
        model,
        x_scaler,
        encoders
    ) -> pd.DataFrame:
        """Sample using DL quantile model."""
        import torch
        from src.utils import sample_paths
        
        # Prepare features
        X_num = X[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
        X_num_scaled = x_scaler.transform(X_num) if x_scaler is not None else X_num.values
        X_num_tensor = np.asarray(X_num_scaled, dtype=np.float32)
        dc_ori_enc = self._encode_with_unknown(encoders['dc_ori'], X['dc_ori'])
        dc_des_enc = self._encode_with_unknown(encoders['dc_des'], X['dc_des'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Predict quantiles
        preds = np.zeros((len(X), len(cfg.DELIVERY_TIME_QUANTILES)), dtype=np.float32)
        bs = 2048
        with torch.no_grad():
            for s in range(0, len(X), bs):
                e = min(len(X), s + bs)
                Xb = torch.tensor(X_num_tensor[s:e], dtype=torch.float32, device=device)
                dc_o = torch.tensor(dc_ori_enc[s:e], dtype=torch.long, device=device)
                dc_d = torch.tensor(dc_des_enc[s:e], dtype=torch.long, device=device)
                preds[s:e] = model(Xb, dc_o, dc_d).cpu().numpy()
        
        # Sample from quantiles
        sampled_labels = np.zeros((len(X), num_replications), dtype=int)
        for i in range(len(X)):
            samples = sample_paths(preds[i], num_replications, rng=rng, lower=0, upper=5)
            bias = float(cfg.DELAY_BIAS_LEVEL)
            if bias != 1.0:
                samples = np.clip(samples * bias, 0.0, 5.0)
            sampled_labels[i] = np.rint(samples).astype(int)
        
        return pd.DataFrame(sampled_labels, index=X.index)
    
    def _sample_catboost(
        self,
        X: pd.DataFrame,
        num_replications: int,
        rng: np.random.Generator,
        model,
        label_encoder
    ) -> pd.DataFrame:
        """Sample using CatBoost classifier."""
        sim_probs = model.predict_proba(X)
        
        # Apply delay bias
        bias_factors = np.array([cfg.DELAY_BIAS_LEVEL ** (i + 1) for i in range(sim_probs.shape[1])])
        sim_probs = sim_probs * bias_factors
        sim_probs = sim_probs / sim_probs.sum(axis=1, keepdims=True)
        
        # Sample
        class_labels = label_encoder.inverse_transform(np.arange(sim_probs.shape[1]))
        rand = rng.random((len(X), num_replications))
        cum_probs = np.cumsum(sim_probs, axis=1)
        sampled_indices = (cum_probs[:, None, :] > rand[..., None]).argmax(axis=2)
        sampled_labels = class_labels[sampled_indices]
        
        return pd.DataFrame(sampled_labels, index=X.index)
    
    def _sample_qrf(
        self,
        X: pd.DataFrame,
        num_replications: int,
        rng: np.random.Generator,
        model
    ) -> pd.DataFrame:
        """Sample using QRF model."""
        from src.utils import sample_paths
        
        preds = model.predict(X, quantiles=cfg.DELIVERY_TIME_QUANTILES)
        
        sampled_labels = np.zeros((len(X), num_replications), dtype=int)
        for i in range(len(X)):
            samples = sample_paths(preds[i], num_replications, rng=rng, lower=0, upper=5)
            sampled_labels[i] = np.rint(samples).astype(int)
        
        return pd.DataFrame(sampled_labels, index=X.index)
