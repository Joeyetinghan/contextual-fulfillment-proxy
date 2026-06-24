# %%
import argparse, numpy as np, torch, optuna
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import src.config as cfg
from src.model.mqrnn import MQRNN
from src.utils import pinball_loss      

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False        
# torch.set_float32_matmul_precision('high')

def plot_loss_curve(loss_tr, loss_val, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(loss_tr,label='train'); plt.plot(loss_val,label='val')
    plt.xlabel('epoch'); plt.ylabel('pinball'); plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(path); plt.close()

def make_fan_dataframe(anchor_dates,              # (N,) datetime64[s]
                       q_hat,                     # (N , Lp , Q)
                       y_hist,                    # (N , Lh)
                       y_future                  # (N , Lp)  ← new
                       ):
    """
    Returns a long DataFrame with one row per time stamp, containing
        date, actual, q_0 … q_{Q-1}
    """
    N, Lp, Q = q_hat.shape
    # --- build horizon datetimes ------------------------------------------------
    step_offsets = np.arange(Lp).astype('timedelta64[h]')      # 1‑hour steps
    horizon_dates = anchor_dates[:, None] + step_offsets[None, :]   # (N,Lp)
    horizon_dates = horizon_dates.reshape(-1)                       # (N*Lp,)

    # --- flatten forecasts & actuals -------------------------------------------
    flat = {f"q_{i}": q_hat[:, :, i].reshape(-1) for i in range(Q)}
    flat["actual"] = y_future.reshape(-1)

    df_fcst = pd.DataFrame({"date": horizon_dates, **flat})

    # --- last historical point so the line connects ----------------------------
    df_hist = pd.DataFrame({
        "date": anchor_dates,
        "actual": y_hist[:, -1]
    })

    return (pd.concat([df_hist, df_fcst], ignore_index=True)
              .sort_values("date")
              .reset_index(drop=True))

def plot_fan(df, quantiles, save_path, title=''):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,4))
    Q         = len(quantiles)
    n_bands   = Q // 2                    # e.g. 11 → 5 bands
    for k in range(n_bands-1, -1, -1):    # k = 4,3,2,1,0
        lo_col = f"q_{k}"                 # 0,1,2,3,4
        hi_col = f"q_{Q-1-k}"             # 10,9,8,7,6
        alpha  = 0.1 + 0.1*(n_bands-k)
        plt.fill_between(df.date,
                         df[lo_col], df[hi_col],
                         step="post", color="grey", alpha=alpha)

    med_idx = quantiles.index(0.5)
    plt.step(df.date, df[f"q_{med_idx}"], where="post",
             color="tab:red", label="median")
    plt.plot(df.date, df.actual, color="black", lw=1, label="actual")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha="right")
    plt.title(title); plt.ylabel("Demand"); plt.xlabel("Period")
    plt.grid(ls="--", alpha=.4); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def time_split(raw: dict[str, np.ndarray],
               split_ratio: float = 0.15) -> tuple[dict, dict]:
    """
    Chronological split: the *earliest* (1‑split_ratio) fraction → train,
    the most‑recent split_ratio fraction → validation.
    Works on the npz‑style dict produced by the feature script.

    Parameters
    ----------
    raw          : {"xh": (N,...), "xp": …, "yh": …, "yp": …, …}
    split_ratio  : float in (0,1) – share for the validation set.

    Returns
    -------
    train_dict, val_dict  – with identical keys, sliced along axis‑0.
    """
    N   = raw["xh"].shape[0]
    cut = int(N * (1.0 - split_ratio))
    train = {k: v[:cut] for k, v in raw.items()}
    val   = {k: v[cut:] for k, v in raw.items()}
    return train, val

# ───────────────────────── Dataset helper ──────────────────────────
class NPZDataset(Dataset):
    def __init__(self, xh, yh, xp, yp, sku, brand):
        self.xh, self.yh, self.xp, self.yp = xh, yh, xp, yp
        self.sku, self.brand = sku, brand
    def __len__(self): return len(self.xh)
    def __getitem__(self, i):
        return ( torch.from_numpy(self.xh[i]).float(),
                 torch.from_numpy(self.yh[i]).float(),
                 torch.from_numpy(self.xp[i]).float(),
                 torch.from_numpy(self.yp[i]).float(),
                 torch.tensor(self.sku[i]).long(),
                 torch.tensor(self.brand[i]).long())
# ──────────────────────── scaling utilities ───────────────────────
def load_npz(path):       # → dict
    arr = np.load(path)
    return {k: arr[k] for k in arr.files if k != 'order_id'}

def fit_scalers(raw):
    hist_flat = raw["xh"].reshape(-1, raw["xh"].shape[-1])
    pred_flat = raw["xp"].reshape(-1, raw["xp"].shape[-1])
    return StandardScaler().fit(hist_flat), StandardScaler().fit(pred_flat)

def apply_scalers(raw, sh, sp):
    xh = sh.transform(raw["xh"].reshape(-1, raw["xh"].shape[-1])
             ).reshape(raw["xh"].shape)
    xp = sp.transform(raw["xp"].reshape(-1, raw["xp"].shape[-1])
             ).reshape(raw["xp"].shape)
    return NPZDataset(xh, raw["yh"], xp, raw["yp"], raw["sku_idx"], raw["brand_idx"])

# ───────────────────────── train / eval loop ───────────────────────
def epoch_loop(model, loader, taus, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xh, yh, xp, yp, sku, brand in loader:
            xh, yh, xp, yp = [t.to(DEV) for t in (xh, yh, xp, yp)]
            sku, brand     = sku.to(DEV), brand.to(DEV)

            q_hat = model(xh, yh, xp, sku, brand)
            loss  = pinball_loss(q_hat, yp, taus)

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            tot += loss.item() * xh.size(0)
            n   += xh.size(0)

    return tot / n

def collect_predictions(model, loader):
    """Collect model predictions and actual values for plotting."""
    model.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for xh, yh, xp, yp, sku, brand in loader:
            xh, yh, xp, yp = [t.to(DEV) for t in (xh, yh, xp, yp)]
            sku, brand = sku.to(DEV), brand.to(DEV)
            
            q_hat = model(xh, yh, xp, sku, brand)
            
            # Take the last prediction step from each batch
            # q_hat shape: (batch_size, forecast_horizon, num_quantiles)
            # yp shape: (batch_size, forecast_horizon)
            all_predictions.append(q_hat[:, -1, :].cpu().numpy())  # Last time step
            all_actuals.append(yp[:, -1].cpu().numpy())  # Last time step
    
    return np.concatenate(all_predictions), np.concatenate(all_actuals)

# ───────────────────────── training wrapper ────────────────────────
def train_model(params, train_set, val_set, meta,
                taus, save_path=None, fixed_epochs=None, return_losses=False):
    is_validation_run = (val_set is not None)
    tr_dl = DataLoader(train_set, batch_size=params["batch_size"],
                       shuffle=True, num_workers=0)
    if is_validation_run:
        vl_dl = DataLoader(val_set, batch_size=4*params["batch_size"],
                           shuffle=False, num_workers=0)

    model = MQRNN(num_cal=meta["num_cal"], num_ord=meta["num_ord"],
                  num_skus=meta["num_skus"], num_brands=meta["num_brands"],
                  sku_emb=params.get("sku_emb", cfg.DEMAND_MODEL_SKU_EMBEDDING_DIM),
                  brand_emb=params.get("brand_emb", cfg.DEMAND_MODEL_BRAND_EMBEDDING_DIM),
                  hidden=params["hidden_dim"], 
                  ctx=params.get("context_dim", cfg.DEMAND_MODEL_CONTEXT_DIM),
                  num_q=len(taus[0,0]),
                  Lp=meta["Lp"], Lh=meta["Lh"],
                  layers=params["lstm_n_layers"],
                  dropout=params["dropout_p"]>0,
                  dropout_p=params["dropout_p"],
                  bidirectional=cfg.DEMAND_MODEL_BIDIRECTIONAL).to(DEV)
    # model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])
    if is_validation_run:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=cfg.DEMAND_MODEL_LR_SCHEDULER_PATIENCE, factor=cfg.DEMAND_MODEL_LR_SCHEDULER_FACTOR)
    
    # Track losses for plotting
    train_losses, val_losses = [], []
    best_val, patience = float("inf"), 0
    E = fixed_epochs or params["epochs"]
    for epoch in range(E):
        train_loss = epoch_loop(model, tr_dl, taus, opt)
        train_losses.append(train_loss)
        
        if is_validation_run:
            val_loss = epoch_loop(model, vl_dl, taus)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val, patience = val_loss, 0
                if save_path: 
                    # Save model with hyperparameters
                    torch.save({
                        'state_dict': model.state_dict(),
                        'hyperparameters': params,
                        'meta': meta,
                        'epoch': epoch + 1
                    }, save_path)
            else:
                patience += 1
                if patience >= params["early_stop"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{E}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{E}, Train Loss: {train_loss:.4f}")
    
    if not is_validation_run and save_path:     # final fit → save
        torch.save({
            'state_dict': model.state_dict(),
            'hyperparameters': params,
            'meta': meta,
            'epoch': E
        }, save_path)
    
    if return_losses:
        return (best_val if is_validation_run else None), train_losses, val_losses
    return best_val if is_validation_run else None

# ───────────────────────────── CLI ─────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=cfg.DEMAND_MODEL_HIDDEN_DIM)
parser.add_argument("--lr", type=float, default=cfg.DEMAND_MODEL_LEARNING_RATE)
parser.add_argument("--quantiles", nargs="+", type=float, default=cfg.DEMAND_MODEL_QUANTILES)
parser.add_argument("--batch_size", type=int, default=cfg.DEMAND_MODEL_BATCH_SIZE)
parser.add_argument("--epochs", type=int, default=cfg.DEMAND_MODEL_EPOCHS)
parser.add_argument("--early_stopping_patience", type=int, default=cfg.DEMAND_MODEL_EARLY_STOPPING_PATIENCE)
parser.add_argument("--lstm_n_layers", type=int, default=cfg.DEMAND_MODEL_LSTM_N_LAYERS)
parser.add_argument("--dropout_p", type=float, default=cfg.DEMAND_MODEL_DROPOUT_P)
parser.add_argument("--weight_decay", type=float, default=cfg.DEMAND_MODEL_WEIGHT_DECAY)
parser.add_argument("--n_trials", type=int, default=0)
parser.add_argument("--models_dir", type=Path, default=cfg.DEMAND_MODELS_DIR)
parser.add_argument('--train_on_proxy', action='store_true', help="Train on forecast+proxy data, evaluate on test data.")
parser.add_argument('--split_ratio', type=float, default=cfg.DEMAND_MODEL_VALIDATION_SPLIT_RATIO)

# Define DEV at module level so it's available when imported
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = parser.parse_args(); args.models_dir.mkdir(parents=True, exist_ok=True)

    # DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taus = torch.tensor(args.quantiles, device=DEV).view(1,1,-1)

    # ─────────────── Load raw arrays (un‑scaled) ───────────────────────
    root = cfg.PROCESSED_DATA_DIR
    forecast_data = load_npz(root/"mqrnn_forecast_train.npz")
    proxy_data    = load_npz(root/"mqrnn_proxy_train.npz")
    test_data     = load_npz(root/"mqrnn_test.npz")

    if args.train_on_proxy:
        print("Training on forecast + proxy data, evaluating on test data.")
        train_raw = {k: np.concatenate([forecast_data[k], proxy_data[k]]) for k in forecast_data}
        train_raw, val_raw = time_split(train_raw, args.split_ratio)
        test_raw = test_data
    else:
        print("Training on forecast data, evaluating on proxy + test data.")
        train_raw = forecast_data
        train_raw, val_raw = time_split(train_raw, args.split_ratio)
        test_raw = {k: np.concatenate([proxy_data[k], test_data[k]]) for k in proxy_data}

    full_raw   = {k: np.concatenate([train_raw[k],
                                     val_raw[k],
                                     test_raw[k]]) for k in train_raw}


    # initial scalers on *train* split
    sc_h, sc_p = fit_scalers(train_raw)
    train_set  = apply_scalers(train_raw, sc_h, sc_p)
    val_set    = apply_scalers(val_raw,   sc_h, sc_p)
    test_set   = apply_scalers(test_raw,  sc_h, sc_p)

    meta = {
        "num_cal" : 4,
        "num_ord" : len(cfg.DEMAND_ORDER_FEATURES),
        "num_skus"  : int(full_raw["sku_idx"].max())   + 1,
        "num_brands": int(full_raw["brand_idx"].max()) + 1,
        "Lh": cfg.DEMAND_MODEL_LOOKBACK,
        "Lp": cfg.DEMAND_FORECAST_HORIZON,
    }

    # ---------------- Hyper‑parameter search ---------------------------
    base = vars(args)
    if args.n_trials > 0:
        SEARCH = cfg.DEMAND_MODEL_HYPERPARAMETER_GRID
        def objective(trial):
            hp = {
                "hidden_dim"    : trial.suggest_categorical("hidden_dim",   SEARCH["hidden_dim"]),
                "lr"            : trial.suggest_float("lr", *SEARCH["lr"], log=True),
                "batch_size"    : trial.suggest_categorical("batch_size",   SEARCH["batch_size"]),
                "lstm_n_layers" : trial.suggest_categorical("lstm_n_layers", SEARCH["lstm_n_layers"]),
                "dropout_p"     : trial.suggest_float("dropout_p", *SEARCH["dropout_p"]),
                "weight_decay"  : trial.suggest_float("weight_decay", *SEARCH["weight_decay"], log=True),
                "context_dim"   : trial.suggest_categorical("context_dim", SEARCH["context_dim"]),
                "sku_emb"       : trial.suggest_categorical("sku_emb", SEARCH["sku_emb"]),
                "brand_emb"     : trial.suggest_categorical("brand_emb", SEARCH["brand_emb"]),
                "early_stop"    : trial.suggest_categorical("early_stopping_patience", SEARCH["early_stopping_patience"]),
                "epochs"        : args.epochs
            }
            best = train_model(hp, train_set, val_set, meta, taus)
            trial.set_user_attr("best_epoch", trial.number)  # placeholder
            return best

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED))
        study.optimize(objective, n_trials=args.n_trials)
        best_hp = study.best_params
        best_epoch = study.best_trial.user_attrs["best_epoch"]
        print("Optuna best hyper‑params:", best_hp)
    else:
        best_hp = {
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "lstm_n_layers": args.lstm_n_layers,
            "dropout_p": args.dropout_p,
            "weight_decay": args.weight_decay,
            "context_dim": cfg.DEMAND_MODEL_CONTEXT_DIM,
            "sku_emb": cfg.DEMAND_MODEL_SKU_EMBEDDING_DIM,
            "brand_emb": cfg.DEMAND_MODEL_BRAND_EMBEDDING_DIM,
            "epochs": args.epochs,
            "early_stop": args.early_stopping_patience
        }
        best_epoch = None                     # will early‑stop

    # ---------------- Final fit on train∪val ----------------------------
    full_raw = {k: np.concatenate([train_raw[k], val_raw[k]])
                for k in train_raw}
    sc_h, sc_p = fit_scalers(full_raw)
    full_set   = apply_scalers(full_raw, sc_h, sc_p)
    train_mode = "_with_proxy" if args.train_on_proxy else ""
    ckpt = args.models_dir/f"mqrnn_model{train_mode}_tuned.pt"

    # Train and collect losses for plotting
    _, train_losses, val_losses = train_model(best_hp, full_set, None, meta, taus, 
                                             save_path=ckpt, fixed_epochs=best_epoch, 
                                             return_losses=True)

    # Plot training curve - commented out to avoid permission errors
    # plot_path = cfg.DEMAND_PLOTS_DIR / f"mqrnn_loss_curve{train_mode}.png"
    # plot_loss_curve(train_losses, val_losses, plot_path)
    # print(f"Loss curve saved to {plot_path}")

    # ---------------- Evaluate on test ----------------------------------
    model = MQRNN(num_cal=meta["num_cal"], num_ord=meta["num_ord"],
                  num_skus=meta["num_skus"], num_brands=meta["num_brands"],
                  sku_emb=best_hp.get("sku_emb", cfg.DEMAND_MODEL_SKU_EMBEDDING_DIM),
                  brand_emb=best_hp.get("brand_emb", cfg.DEMAND_MODEL_BRAND_EMBEDDING_DIM),
                  hidden=best_hp["hidden_dim"], 
                  ctx=best_hp.get("context_dim", cfg.DEMAND_MODEL_CONTEXT_DIM),
                  num_q=len(taus[0,0]), Lp=meta["Lp"], Lh=meta["Lh"],
                  layers=best_hp["lstm_n_layers"],
                  dropout=best_hp["dropout_p"]>0,
                  dropout_p=best_hp["dropout_p"],
                  bidirectional=cfg.DEMAND_MODEL_BIDIRECTIONAL).to(DEV)

    # Load checkpoint with hyperparameters
    checkpoint = torch.load(ckpt, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")


    test_loader = DataLoader(test_set, batch_size=4*best_hp["batch_size"],
                             shuffle=False, num_workers=0)
    test_loss = epoch_loop(model, test_loader, taus)
    print(f"Test pinball loss: {test_loss:.4f}")

    # Collect predictions and plot forecast vs actual - commented out to avoid permission errors
    # print("Generating forecast vs actual plot...")
    # q_hat_test, y_test = collect_predictions(model, test_loader)

    # # Get corresponding dates for the test predictions
    # test_dates = test_raw["dt"]

    # unique_skus = np.unique(test_set.sku)    # integer indices
    # from joblib import Parallel, delayed

    # def process_sku(sku_id, test_set, test_dates, model, args, DEV):
    #     """Process a single SKU for fan chart generation."""
    #     mask = test_set.sku == sku_id
    #     anchor_dates = test_dates[mask]                  # (N,)
    #     y_hist       = test_set.yh[mask]                  # (N,Lh)
    #     y_future     = test_set.yp[mask]                  # (N,Lp)
    #     xh = test_set.xh[mask]
    #     xp = test_set.xp[mask]
    #     sku = test_set.sku[mask]
    #     brand = test_set.brand[mask]
    #     yh = test_set.yh[mask]
    #     with torch.no_grad():
    #         q_hat = model(torch.tensor(xh).to(DEV), torch.tensor(yh).to(DEV), torch.tensor(xp).to(DEV),
    #                     torch.tensor(sku).to(DEV), torch.tensor(brand).to(DEV)).cpu().numpy()

    #     df = make_fan_dataframe(anchor_dates,
    #                             q_hat,
    #                             y_hist,
    #                             y_future)

    #     plot_fan(df, args.quantiles,
    #             save_path=cfg.DEMAND_PLOTS_DIR/f'skus/{sku_id}_fan{train_mode}.png',
    #             title=f'SKU {sku_id}')

    # Parallel(n_jobs=-1)(delayed(process_sku)(sku_id, test_set, test_dates, model, args, DEV) 
    #                    for sku_id in unique_skus)

    # save scalers
    torch.save({"scaler_hist": sc_h, "scaler_pred": sc_p},
               args.models_dir/f"mqrnn_scalers{train_mode}_tuned.pt")
    print(f"Checkpoint + scalers saved to {args.models_dir}")
