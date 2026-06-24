import numpy as np
import torch
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
import optuna
import src.config as cfg
from src.training.delivery_time.common import prepare_categorical_encoders
from src.training.delivery_time._train_dl import train_dl_model


def train_catboost_simulator(df_train, n_trials=0, carrier_id=None):
    """
    Trains a calibrated CatBoost classifier to simulate delivery times.
    """
    model_name = "Global Delivery Simulator"
    if carrier_id:
        model_name = f"Delivery Simulator for Carrier {carrier_id}"
    print(f"\n--- Training {model_name} ---")

    # Define features and target
    y = df_train['delivery_time_days']

    # Filter out classes with too few samples for reliable cross-validation
    class_counts = y.value_counts()
    min_samples_per_class = 10  # A reasonable number for 3-fold CV
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    
    if len(valid_classes) < 2:
        print("Skipping training: Not enough classes with sufficient samples in the dataset.")
        return None, None
            
    df_filtered = df_train[y.isin(valid_classes)].copy()
    y = df_filtered['delivery_time_days']
    
    features = cfg.DELIVERY_TIME_FEATURES
    X = df_filtered[features].fillna(0)


    print(f"Training on {len(X)} samples for {len(valid_classes)} classes.")

    # Encode target labels to be in [0, n_classes-1] for the classifier
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Calculate class weights for handling imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weights_dict = dict(zip(np.unique(y_encoded), class_weights))

    if n_trials > 0:
        # --- Optuna Hyperparameter Tuning ---
        default_params = {
            'iterations': 1000, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3.0,
            'bagging_temperature': 1.0, 'random_strength': 1.0, 
            'colsample_bylevel': 1.0, 'od_wait': 20
        }

        def objective(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 500, 1200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'logging_level': 'Silent',
                'od_type': 'Iter',
                'od_wait': trial.suggest_int('od_wait', 20, 50),
                'random_state': cfg.RANDOM_SEED,
                'class_weights': class_weights_dict,
                'cat_features': cfg.DELIVERY_DL_CATEGORICAL_FEATURES
            }
            model = CatBoostClassifier(**param, allow_writing_files=False)
            scores = cross_val_score(model, X, y_encoded, cv=3, scoring='neg_log_loss', n_jobs=-1)
            return scores.mean()
        
        print(f"Starting hyperparameter tuning ({n_trials} trials)...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED))
        study.enqueue_trial(default_params)
        study.optimize(objective, n_trials=n_trials)
        print(f"Tuning complete. Best neg_log_loss: {study.best_value:.4f}")
        
        best_params = study.best_params
        best_params['loss_function'] = 'MultiClass'
        best_params['random_state'] = cfg.RANDOM_SEED
        best_params['verbose'] = 0
        best_params['class_weights'] = class_weights_dict
        best_params['cat_features'] = cfg.DELIVERY_DL_CATEGORICAL_FEATURES
        catboost_model = CatBoostClassifier(**best_params, allow_writing_files=False)

    else:
        # Use default parameters if not tuning
        catboost_model = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function='MultiClass', class_weights=class_weights_dict,
            random_state=cfg.RANDOM_SEED, verbose=0, allow_writing_files=False,
            cat_features=cfg.DELIVERY_DL_CATEGORICAL_FEATURES
        )

    # Calibrate the classifier using isotonic regression with 3-fold CV
    calibrated_clf = CalibratedClassifierCV(
        estimator=catboost_model,
        method='isotonic',
        cv=3,
        n_jobs=-1
    )

    print(f"Training and calibrating simulator...")
    calibrated_clf.fit(X, y_encoded)
    print("Training complete.")
    
    return calibrated_clf, le


def train_dl_simulator(df_train, n_trials: int = 0):
    """
    Train a DL quantile model on the TEST set to serve as a simulator-oracle.
    Saves a torch bundle and a small joblib pointer with type='dl'.
    """
    print("--- Training DL Simulator on TEST set ---")
    cfg.DELIVERY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    features = cfg.DELIVERY_TIME_FEATURES
    df = df_train.dropna(subset=[cfg.DELIVERY_TIME_TARGET]).reset_index(drop=True)
    X = df[features].fillna(0)
    y = df[cfg.DELIVERY_TIME_TARGET]

    encoders, vocab_sizes = prepare_categorical_encoders(df)
    X_num = X[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    x_scaler = StandardScaler().fit(X_num)
    X_num_scaled = x_scaler.transform(X_num)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xn = torch.tensor(X_num_scaled, dtype=torch.float32)
    Xc = {}
    for cat in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
        if cat in X.columns:
            Xc[cat] = torch.tensor(encoders[cat].transform(X[cat].astype(str)), dtype=torch.long)
    yt = torch.tensor(y.values, dtype=torch.float32)

    split = int(0.8 * len(Xn))
    if n_trials > 0:
        def objective(trial):
            params = {
                'lr': trial.suggest_float('lr', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['lr'], log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['hidden_dim']),
                'n_layers': trial.suggest_categorical('n_layers', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['n_layers']),
                'dropout_p': trial.suggest_float('dropout_p', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['dropout_p']),
                'weight_decay': trial.suggest_float('weight_decay', *cfg.DELIVERY_DL_HYPERPARAMETER_GRID['weight_decay'], log=True),
                'batch_size': trial.suggest_categorical('batch_size', cfg.DELIVERY_DL_HYPERPARAMETER_GRID['batch_size']),
                'epochs': cfg.DELIVERY_DL_EPOCHS,
            }
            return train_dl_model(
                params,
                Xn[:split], 
                {'dc_ori': Xc['dc_ori'][:split], 'dc_des': Xc['dc_des'][:split]}, 
                yt[:split],
                Xn[split:], 
                {'dc_ori': Xc['dc_ori'][split:], 'dc_des': Xc['dc_des'][split:]}, 
                yt[split:],
                vocab_sizes, 
                device, 
                trial=trial
            )

        print(f"Starting hyperparameter tuning for DL simulator ({n_trials} trials)...")
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_epoch = study.best_trial.user_attrs.get('best_epoch', cfg.DELIVERY_DL_EPOCHS)
        best_params['epochs'] = best_epoch
        model, _, _ = train_dl_model(
            best_params,
            Xn[:split], 
            {'dc_ori': Xc['dc_ori'][:split], 'dc_des': Xc['dc_des'][:split]}, 
            yt[:split],
            Xn[split:], 
            {'dc_ori': Xc['dc_ori'][split:], 'dc_des': Xc['dc_des'][split:]}, 
            yt[split:],
            vocab_sizes, 
            device, 
            return_model=True
        )
        final_params = best_params
    else:
        params = {
            'lr': 1e-3,
            'hidden_dim': 128,
            'n_layers': 4,
            'dropout_p': 0.15,
            'weight_decay': 1e-3,
            'batch_size': 128,
            'epochs': cfg.DELIVERY_DL_EPOCHS,
        }
        model, _, _ = train_dl_model(
            params,
            Xn[:split], 
            {k: v[:split] for k, v in Xc.items()}, 
            yt[:split],
            Xn[split:], 
            {k: v[split:] for k, v in Xc.items()}, 
            yt[split:],
            vocab_sizes, 
            device, 
            return_model=True
        )
        final_params = params
    
    return model, x_scaler, encoders, vocab_sizes, final_params, Xn, Xc, y
