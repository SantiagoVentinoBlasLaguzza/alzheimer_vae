#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep-ready training + evaluation para β-VAE + clasificador rápido
Optimized for W&B sweeps targeting AUC from post-training CV.
"""

from __future__ import annotations
import os
import sys
import random
import math
import warnings
import time
from collections import Counter
from typing import Dict, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
VAR_TH = 1e-2  # umbral de varianza mínima para conservar una dimensión
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv  # ← Agregá esta línea primero
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold  # ⬅️ NUEVO
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve  # ← añade roc_curve


# ── aceleradores PyTorch 2 ──────────────────────────────
import torch.backends.cuda as torch_back
torch_back.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ────────────────────── Configuración de PATH e import condicional ──────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")
light_pipeline_path = os.path.join(script_dir, 'light_pipeline.py')
print(f"Checking for light_pipeline at: {light_pipeline_path}")
print(f"Does file exist? {os.path.exists(light_pipeline_path)}")

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    print(f"Added {script_dir} to sys.path")

try:
    from light_pipeline import extract_latents, clinical_onehot, run_cv
    LIGHT_PIPELINE_AVAILABLE = True
    print("Successfully imported from light_pipeline.py after path modification")
except ImportError as e:
    LIGHT_PIPELINE_AVAILABLE = False
    print(f"Warning: Still could not import light_pipeline.py. Error: {e}")
    print(f"Current sys.path: {sys.path}")

    def extract_latents(*args, **kwargs):
        raise NotImplementedError("light_pipeline not available")

    def clinical_onehot(*args, **kwargs):
        raise NotImplementedError("light_pipeline not available")

    def run_cv(*args, **kwargs):
        raise NotImplementedError("light_pipeline not available")

# ────────────────────── Configuration & Constants ──────────────────────
PROJECT_DIR: str = "/content/drive/MyDrive/GrandMeanNorm"
FOLD: int = 1
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED: int = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ────────────────────── W&B Sweep Defaults ──────────────────────
DEFAULTS: Dict[str, Any] = dict(
    lr=1e-4,
    batch_size=64,
    beta=25,
    beta_ramp_epochs=400,
    epochs=450,
    k_filters=32,
    latent_dim=128,
    eval_interval=25,
    patience=150,
    VAR_TH = 1e-3  # Umbral de varianza mínima para conservar una dimensión latente
)

# ────────────────────── Data Loading Utilities ─────────────────────────
def _load(fdir: str, name: str) -> Any:
    """Helper to load torch files with error handling."""
    filepath = os.path.join(fdir, name)
    try:
        return torch.load(filepath, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        raise

def load_fold_full(fold: int = 1) -> Tuple[torch.Tensor, ...]:
    """Loads all data splits and labels for a given fold."""
    fdir = f"{PROJECT_DIR}/fold_{fold}"
    print(f"Loading data from: {fdir}")
    trX, vaX, teX = (_load(fdir, f"{p}_z.pt") for p in ("train", "val", "test"))
    trL, vaL, teL = (_load(fdir, f"{p}_labels.pt") for p in ("train", "val", "test"))

    # Encode labels: AD=1, CN=0, Other=2
    def lab(s): return 1 if isinstance(s, str) and s.startswith("AD_") else (0 if isinstance(s, str) and s.startswith("CN_") else 2)
    trY, vaY, teY = (
        torch.tensor([lab(s) for s in lst], dtype=torch.long)
        for lst in (trL, vaL, teL)
    )

    print("Data shapes:")
    print(f"  Train X: {trX.shape}, Y: {trY.shape}")
    print(f"  Val   X: {vaX.shape}, Y: {vaY.shape}")
    print(f"  Test  X: {teX.shape}, Y: {teY.shape}")
    print("Label counts (0:CN, 1:AD, 2:Other):")
    print(f"  Train: {Counter(trY.numpy())}")
    print(f"  Val:   {Counter(vaY.numpy())}")
    print(f"  Test:  {Counter(teY.numpy())}")

    return trX, vaX, teX, trL, vaL, teL, trY, vaY, teY
# ────────────────────── β-VAE Architecture ─────────────────────────────
class Encoder(nn.Module):
    def __init__(self, c_in: int, latent: int, k: int = 32):
        super().__init__()
        print(f"Initializing Encoder: c_in={c_in}, latent={latent}, k={k}")
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),
            nn.Conv2d(k, 2*k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),
            nn.Conv2d(2*k, 4*k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),
            nn.Conv2d(4*k, 8*k, kernel_size=3, stride=2, padding=0), nn.LeakyReLU(0.1),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, c_in, 166, 166)
            conv_output_shape = self.conv(dummy_input).shape
            flattened_size = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]
            print(f"  Calculated flattened size: {flattened_size} (Shape: {conv_output_shape})")
        self.flat = nn.Flatten()
        fc_intermediate_size = flattened_size // 10
        self.fc = nn.Sequential(nn.Linear(flattened_size, fc_intermediate_size), nn.LeakyReLU(0.1))
        self.mu = nn.Linear(fc_intermediate_size, latent)
        self.logv = nn.Linear(fc_intermediate_size, latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_conv = self.conv(x)
        h_flat = self.flat(h_conv)
        h_fc = self.fc(h_flat)
        return self.mu(h_fc), self.logv(h_fc)

class Decoder(nn.Module):
    def __init__(self, latent: int, c_out: int, k: int = 32):
        super().__init__()
        print(f"Initializing Decoder: latent={latent}, c_out={c_out}, k={k}")
        self.k8 = 8 * k
        self.reshape_dims = (self.k8, 9, 9)
        fc_output_size = self.k8 * 9 * 9
        print(f"  Decoder FC output size: {fc_output_size}, Reshape dims: {self.reshape_dims}")
        self.fc = nn.Sequential(nn.Linear(latent, fc_output_size), nn.LeakyReLU(0.1))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.k8, 4*k, kernel_size=3, stride=2, padding=0), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(4*k, 2*k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(2*k, k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(k, c_out, kernel_size=4, stride=2, padding=0)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h_fc = self.fc(z)
        h_reshaped = h_fc.view(-1, *self.reshape_dims)
        return self.deconv(h_reshaped)

class BetaVAE(nn.Module):
    def __init__(self, c_in: int, latent: int, k: int):
        super().__init__()
        self.enc = Encoder(c_in, latent, k)
        self.dec = Decoder(latent, c_in, k)

    def _reparam(self, mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logv = self.enc(x)
        z = self._reparam(mu, logv)
        x_hat = self.dec(z)
        return x_hat, mu, logv

def loss_vae(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logv: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1)
    kld = torch.mean(kld)
    total_loss = recon_loss + beta * kld
    return total_loss, recon_loss, kld

@torch.no_grad()
def encode_mu(encoder: nn.Module, X: torch.Tensor, bs: int = 256) -> np.ndarray:
    encoder.eval()
    Z_list = []
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, pin_memory=True)
    for (x_batch,) in loader:
        x_batch = x_batch.to(DEVICE, memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(DEVICE.type == 'cuda')):
            mu, _ = encoder(x_batch)
        Z_list.append(mu.cpu().float().numpy())
    return np.vstack(Z_list)

# ────────────────────── VAE Training Function
def train_vae(cfg: wandb.Config,
              trX: torch.Tensor, vaX: torch.Tensor,
              trY: torch.Tensor, vaY: torch.Tensor) -> Tuple[BetaVAE, Dict[str, Any]]:
    """Trains the Beta-VAE model and returns the trained model and best state."""
    print("\n--- Starting VAE Training ---")
    print(f"Using device: {DEVICE}")
    print("Sweep Config:")
    for key, val in cfg.items():
        print(f" {key}: {val}")

    # --- Data Preparation ---
    mask_tr_adcn = trY != 2
    mask_va_adcn = vaY != 2
    trX_adcn, trY_adcn_np = trX[mask_tr_adcn], trY[mask_tr_adcn].numpy()
    vaX_adcn, vaY_adcn_np = vaX[mask_va_adcn], vaY[mask_va_adcn].numpy()

    print("\nData for Intermediate Evaluation (AD/CN only):")
    print(f" Train AD/CN X shape: {trX_adcn.shape}, Y shape: {trY_adcn_np.shape}")
    print(f" Val AD/CN X shape: {vaX_adcn.shape}, Y shape: {vaY_adcn_np.shape}")

    #vae_train_X = torch.cat([trX, vaX])
    vae_train_X = trX.clone()
    print(f"\nVAE Training Data shape (Train+Val): {vae_train_X.shape}")
    vae_train_dataset = TensorDataset(vae_train_X)
    train_loader = DataLoader(
        vae_train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=min(4, os.cpu_count()),
        persistent_workers=min(4, os.cpu_count()) > 0,
        prefetch_factor=4 if min(4, os.cpu_count()) > 0 else None,
        drop_last=True,
    )

    val_recon_dataset = TensorDataset(vaX)
    val_recon_loader = DataLoader(val_recon_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

    C = trX.shape[1]
    vae = BetaVAE(C, cfg.latent_dim, cfg.k_filters).to(DEVICE, memory_format=torch.channels_last)
    print(f"Compiling model with mode: reduce-overhead")
    vae = torch.compile(vae, mode="reduce-overhead")

    wandb.watch(vae, log="gradients", log_freq=100)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.lr, fused=(DEVICE.type == 'cuda'))

    use_amp = DEVICE.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP Enabled: {use_amp}, dtype: {amp_dtype}")

    best_val_recon_loss = math.inf
    epochs_no_improve = 0
    best_state_dict = None

    eval_interval = getattr(cfg, "eval_interval", DEFAULTS['eval_interval'])
    patience = getattr(cfg, "patience", DEFAULTS['patience'])

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start_time = time.time()
        vae.train()
        total_loss, total_recon, total_kld = 0.0, 0.0, 0.0

        for i, (x_batch,) in enumerate(train_loader):
            x_batch = x_batch.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                x_hat, mu, logv = vae(x_batch)
                beta_t = min(cfg.beta, cfg.beta * (epoch / cfg.beta_ramp_epochs)) if cfg.beta_ramp_epochs > 0 else cfg.beta
                loss, recon, kld = loss_vae(x_hat, x_batch, mu, logv, beta_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()

        num_batches = len(train_loader)
        avg_train_loss = total_loss / num_batches
        avg_train_recon = total_recon / num_batches
        avg_train_kld = total_kld / num_batches

        vae.eval()
        val_recon_loss = 0.0
        with torch.no_grad():
            for (x_val_batch,) in val_recon_loader:
                x_val_batch = x_val_batch.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                    x_val_hat, mu_val, logv_val = vae(x_val_batch)
                    _, v_recon, _ = loss_vae(x_val_hat, x_val_batch, mu_val, logv_val, cfg.beta)
                val_recon_loss += v_recon.item()
        avg_val_recon_loss = val_recon_loss / len(val_recon_loader)
        epoch_time = time.time() - epoch_start_time

        log_dict = {
            "epoch": epoch,
            "train_loss_epoch": avg_train_loss,
            "train_recon_epoch": avg_train_recon,
            "train_kld_epoch": avg_train_kld,
            "val_recon_loss_epoch": avg_val_recon_loss,
            "beta_effective": beta_t,
            "epoch_time_sec": epoch_time,
            "gpu_mem_alloc_gb": torch.cuda.memory_allocated(DEVICE) / (1024**3) if DEVICE.type == 'cuda' else 0,
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved(DEVICE) / (1024**3) if DEVICE.type == 'cuda' else 0,
        }

        if eval_interval > 0 and (epoch % eval_interval == 0 or epoch == 1 or epoch == cfg.epochs):
            acc_tr, auc_tr, acc_va, auc_va = quick_metrics(vae.enc, trX_adcn, trY_adcn_np, vaX_adcn, vaY_adcn_np)
            log_dict.update({
                "ACC_train_mu_quick": acc_tr,
                "AUC_train_mu_quick": auc_tr,
                "ACC_val_mu_quick": acc_va,
                "AUC_val_mu_quick": auc_va,
            })
            wandb.summary["VAL_LogReg_AUC"] = auc_va
            wandb.summary["VAL_LogReg_AUC_dict"] = {"fold1": auc_va}

        print("Logging dict to W&B:", log_dict)
        try:
            wandb.log(log_dict)
        except Exception as e:
            print(f"[W&B LOGGING ERROR] {e}")

        print(f"Epoch {epoch}/{cfg.epochs} [{epoch_time:.2f}s] - "
              f"Train Loss: {avg_train_loss:.2f} (Recon: {avg_train_recon:.2f}, KLD: {avg_train_kld:.2f}) | "
              f"Val Recon Loss: {avg_val_recon_loss:.2f} | Beta: {beta_t:.2f}")

        tolerance = 1e-4
        if avg_val_recon_loss < best_val_recon_loss - tolerance:
            print(f" Validation reconstruction loss improved ({best_val_recon_loss:.4f} -> {avg_val_recon_loss:.4f}). Saving model state.")
            best_val_recon_loss = avg_val_recon_loss
            epochs_no_improve = 0
            best_state_dict = {k: v.detach().clone().cpu() for k, v in vae.state_dict().items()}
        else:
            epochs_no_improve += 1
            print(f" Validation reconstruction loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    total_training_time = time.time() - start_time
    print(f"\n--- VAE Training Finished ---")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Stopped at epoch: {epoch}")

    if best_state_dict:
        print(f"Loading best model state (Val Recon Loss: {best_val_recon_loss:.4f}) from epoch {epoch - epochs_no_improve}")
        vae.load_state_dict(best_state_dict)
    else:
        print("Warning: No improvement observed or early stopping patience was 0. Using model from final epoch.")

    wandb.summary["best_val_recon_loss"] = best_val_recon_loss
    wandb.summary["stopped_epoch"] = epoch
    wandb.summary["total_training_time_min"] = total_training_time / 60

    return vae, best_state_dict


# ─────────────────────── quick_metrics mejorado ───────────────────────
def quick_metrics(
    encoder: nn.Module,
    X_train: torch.Tensor, y_train: np.ndarray,
    X_val:   torch.Tensor, y_val:   np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    μ-features  →  z-score  →  filtro var<VAR_TH  →  Logistic Regression
    Umbral optimizado (Youden J). Devuelve ACC/AUC train y val.
    """
    t0 = time.time()
    try:
        # --- 1. codificar μ ---
        Z_tr = encode_mu(encoder, X_train)
        Z_va = encode_mu(encoder, X_val)

        # --- 2. estandarizar ---
        scaler = StandardScaler()
        Z_tr = scaler.fit_transform(Z_tr)
        Z_va = scaler.transform(Z_va)

        # --- 3. filtrar dims con varianza muy baja ---
        keep = np.var(Z_tr, axis=0) > VAR_TH  # usa el mismo VAR_TH global
        Z_tr, Z_va = Z_tr[:, keep], Z_va[:, keep]


        # --- 4. entrenar LogReg balanceada ---
        clf = LogisticRegression(max_iter=4000, class_weight="balanced",
                                 random_state=SEED, solver="liblinear")
        clf.fit(Z_tr, y_train)

        p_tr = clf.predict_proba(Z_tr)[:, 1]
        p_va = clf.predict_proba(Z_va)[:, 1]

        # --- 5. optimizar umbral (Youden J) ---
        fpr, tpr, thr = roc_curve(y_train, p_tr)
        best_thr = thr[np.argmax(tpr - fpr)]

        # --- 6. métricas ---
        acc_tr = accuracy_score(y_train, p_tr >= best_thr)
        auc_tr = roc_auc_score(y_train,  p_tr)
        acc_va = accuracy_score(y_val,   p_va >= best_thr)
        auc_va = roc_auc_score(y_val,    p_va)

        print(f"Quick metrics {time.time()-t0:.1f}s | "
              f"Train AUC {auc_tr:.3f} ACC {acc_tr:.3f} | "
              f"Val AUC {auc_va:.3f} ACC {acc_va:.3f} "
              f"(dims kept {keep.sum()}/{keep.size})")

        # Opcional: log a W&B
        if "wandb" in sys.modules:
            wandb.log({"QuickMetrics_ValAUC": auc_va, "QuickMetrics_ValACC": acc_va})

        return acc_tr, auc_tr, acc_va, auc_va

    except Exception as e:
        print("‼️ quick_metrics error:", e)
        return 0.0, 0.0, 0.0, 0.0

    
# ────────────────── Final Classifier Evaluation (using light_pipeline) ─────
def evaluate_classifiers(
    encoder: nn.Module,
    trX: torch.Tensor, vaX: torch.Tensor,
    trL: list, vaL: list,
    trY: torch.Tensor, vaY: torch.Tensor
) -> None:
    """
    Performs detailed classifier evaluation using functions imported from
    light_pipeline.py. Logs results to W&B.
    Moves encoder to CPU for evaluation.
    """
    if not LIGHT_PIPELINE_AVAILABLE:
        print("\nSkipping final classifier evaluation because light_pipeline functions are not available.")
        wandb.log({"AUC_test_mu": 0.0})
        wandb.summary["AUC_test_mu"] = 0.0
        return

    print("\n--- Starting Final Classifier Evaluation (using light_pipeline) ---")
    start_time = time.time()

    encoder.eval()
    encoder_cpu = encoder.to('cpu')
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    print("Encoder moved to CPU for evaluation.")

    X_full = torch.cat([trX, vaX]).cpu()
    y_full = torch.cat([trY, vaY]).cpu().numpy()
    ids_full = np.array(list(trL) + list(vaL))

    mask_adcn = y_full != 2
    X_adcn = X_full[mask_adcn]
    y_adcn = y_full[mask_adcn]
    ids_adcn = ids_full[mask_adcn]
    print(f"Data for CV (AD/CN only): X shape {X_adcn.shape}, Y shape {y_adcn.shape}")
    print(f"Class distribution for CV: {Counter(y_adcn)}")

    try:
        clinical_features = clinical_onehot(ids_adcn)
        print(f"Clinical features shape: {clinical_features.shape}")
    except Exception as e:
        print(f"Warning: Could not generate clinical features: {e}. Proceeding without them.")
        clinical_features = None

    final_sweep_metric = 0.0

    for use_sigma, tag in [(False, "mu"), (True, "mu_sigma")]:
        print(f"\nEvaluating classifiers for feature type: '{tag}'")
        try:
            Z_features = extract_latents(encoder_cpu, X_adcn, with_sigma=use_sigma, bs=256)
            print(f" Latent features ({tag}) shape: {Z_features.shape}")

            results_df = run_cv(Z_features, y_adcn, clinical=clinical_features, tag=tag)

            if results_df is not None and not results_df.empty:
                mean_results = results_df.groupby("model")[["auc", "bal"]].mean()
                print(f"\nMean CV Results ({tag}):")
                print(mean_results)
                # después de mean_results = ...
                if not use_sigma and "LogReg" in mean_results.index:
                    auc_mu = mean_results.loc["LogReg", "auc"]
                    wandb.summary["CV_mu_LogReg_AUC_mean"] = auc_mu      #  <-- añadir

                if pd.isna(auc_mu):              # ← justo después de calcular auc_mu
                    auc_mu = 0.0


                for model_name, row in mean_results.iterrows():
                    wandb.log({
                        f"CV_{tag}_{model_name}_AUC_mean": row.auc,
                        f"CV_{tag}_{model_name}_BAL_mean": row.bal
                    })
                    wandb.summary[f"CV_{tag}_{model_name}_AUC_mean"] = row.auc

                if not use_sigma:
                    if "LogReg" in mean_results.index:
                        auc_test_mu = mean_results.loc["LogReg", "auc"]
                        if pd.isna(auc_test_mu):
                            print("Warning: AUC for LogReg ('mu') is NaN. Logging 0.")
                            auc_test_mu = 0.0
                        print(f"\nLogging sweep target metric: AUC_test_mu = {auc_test_mu:.4f}")
                        final_sweep_metric = auc_test_mu
                    else:
                        print("Warning: Logistic Regression results not found in CV output for 'mu' features.")
                        final_sweep_metric = 0.0
            else:
                print(f"Warning: CV results DataFrame for tag '{tag}' is None or empty. Skipping logging.")
                if not use_sigma:
                    final_sweep_metric = 0.0

        except Exception as e:
            print(f"Error during classifier evaluation for tag '{tag}': {e}")
            if not use_sigma:
                final_sweep_metric = 0.0

    eval_time = time.time() - start_time
    print(f"\n--- Classifier Evaluation Finished ---")
    print(f"Total evaluation time: {eval_time:.2f} seconds ({eval_time / 60:.2f} minutes)")

    print(f"Final value being logged for AUC_test_mu: {final_sweep_metric}")
    wandb.log({"AUC_test_mu": final_sweep_metric})
    wandb.summary["AUC_test_mu"] = final_sweep_metric
# ────────────────── Test-set evaluation helpers ──────────────────
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def evaluate_on_test(
    encoder: nn.Module,
    trX: torch.Tensor, vaX: torch.Tensor, teX: torch.Tensor,
    trY: torch.Tensor, vaY: torch.Tensor, teY: torch.Tensor,
    *,
    classifiers: Optional[Dict[str, Any]] = None,
    wandb_step_prefix: str = "TEST_",
) -> None:
    """
    ♦ Entrena los clasificadores definidos en `classifiers`
    ♦ Con X_train = trX ∪ vaX , y_train = trY ∪ vaY (solo AD/CN)
    ♦ Evalúa en X_test = teX (solo AD/CN)
    ♦ Loggea todas las métricas en W&B
    """

    # -- 1. preparar datos AD/CN -----------------------------------
    mask_trva = torch.cat([trY, vaY]) != 2
    mask_te = teY != 2

    X_train = torch.cat([trX, vaX])[mask_trva]
    y_train = torch.cat([trY, vaY])[mask_trva].numpy()
    X_test = teX[mask_te]
    y_test = teY[mask_te].numpy()

    if len(np.unique(y_test)) < 2:
        print("⚠️ Test set no contiene ambas clases AD y CN. Se omite evaluación.")
        return

    # -- 2. extraer latentes ---------------------------------------
    encoder.eval()
    encoder_cpu = encoder.to("cpu")

    Z_train = extract_latents(encoder_cpu, X_train, with_sigma=False, bs=256)
    Z_test = extract_latents(encoder_cpu, X_test, with_sigma=False, bs=256)

    # -- 3. definir clasificadores por defecto ---------------------
    if classifiers is None:
        classifiers = {
            # modelo            , espacio de parámetros                   , usa_proba?
            "LogReg": (
                LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=4000, random_state=SEED),
                {"C": np.logspace(-2, 1, 10)},
                True,
            ),
            "SVM_RBF": (
                SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
                {"C": np.logspace(-1, 2, 8), "gamma": ["scale", "auto"]},
                True,
            ),
            "RF": (
                RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
                {"n_estimators": [100, 300, 600], "max_depth": [None, 10, 20]},
                True,
            ),
            "GB": (
                GradientBoostingClassifier(random_state=SEED),
                {"n_estimators": [100, 300], "learning_rate": [0.01, 0.05, 0.1]},
                True,
            ),
        }


    # -- 4. escalar y filtrar varianza nula ------------------------
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z_train)
    Z_test = scaler.transform(Z_test)

    var_mask = np.std(Z_train, axis=0) > 0
    Z_train = Z_train[:, var_mask]
    Z_test = Z_test[:, var_mask]

    # -- 5. entrenar / evaluar -------------------------------------
    # -- 5. entrenar / evaluar con búsqueda sucesiva ---------------
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    for name, (base_clf, param_grid, has_proba) in classifiers.items():
        search = HalvingRandomSearchCV(
            base_clf,
            param_grid,
            factor=3,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            random_state=SEED,
            verbose=0,
        )
        search.fit(Z_train, y_train)
        clf = search.best_estimator_

        # ---------------- Predicciones -----------------
        probas = (clf.predict_proba(Z_test)[:, 1] if has_proba
                else clf.decision_function(Z_test))
        preds = (probas >= 0.5).astype(int)

        # ---------------- Métricas ---------------------
        auc = roc_auc_score(y_test, probas)
        bal = balanced_accuracy_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)


        # -- 6. loggear --------------------------------------------
        wandb.log({
            f"{wandb_step_prefix}{name}_AUC": auc,
            f"{wandb_step_prefix}{name}_BAL": bal,
            f"{wandb_step_prefix}{name}_ACC": acc,
            f"{wandb_step_prefix}{name}_F1": f1,
        })
        wandb.summary[f"{wandb_step_prefix}{name}_AUC"] = auc
        print(f"[TEST] {name:8s} | AUC={auc:.3f} BAL={bal:.3f} ACC={acc:.3f} F1={f1:.3f}")

# ────────────────────── Main
def main():
    """Main function to initialize W&B, run training, and evaluation."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    wandb.init(config=DEFAULTS, reinit=True)
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    # en main(), después de otros defines
    wandb.define_metric("CV_mu_LogReg_AUC_mean", summary="max")
    # Definir métricas finales de test explícitamente en el dashboard
    for model in ["LogReg", "SVM_RBF", "RF", "GB"]:
        wandb.define_metric(f"TEST_{model}_AUC", summary="max")
        wandb.define_metric(f"TEST_{model}_BAL", summary="max")
        wandb.define_metric(f"TEST_{model}_ACC", summary="max")
        wandb.define_metric(f"TEST_{model}_F1",  summary="max")

    #wandb.define_metric("VAL_LogReg_AUC", summary="max")

    cfg = wandb.config
    print("W&B Run Config:", cfg)

    data_load_start = time.time()
    trX, vaX, teX, trL, vaL, teL, trY, vaY, teY = load_fold_full(FOLD)
    print(f"Data loading took: {time.time() - data_load_start:.2f}s")

    vae_model, _ = train_vae(cfg, trX, vaX, trY, vaY)

    # ──────────────────────────────────────────────────────────────
    # NUEVO: 5×4 Nested-CV sobre μ (y μ+σ) ⇒ guarda CV_mu_LogReg_AUC_mean
    # ──────────────────────────────────────────────────────────────
    evaluate_classifiers(
        vae_model.enc,
        trX, vaX,
        trL, vaL,
        trY, vaY
    )

    evaluate_on_test(
        vae_model.enc,
        trX, vaX, teX,
        trY, vaY, teY,
    )

    wandb.finish()
    print("\n--- W&B Run Finished ---")


if __name__ == "__main__":
    import pandas as pd
    main()
