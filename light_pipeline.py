from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal utilities from the original *light_pipeline*:

• clinical_onehot
• extract_latents
• run_cv

The rest (data loading, encoder definition, main(), etc.) has been
removed to keep the file focused on the three reusable functions that
hyperparameters.py necesita. Only helper code strictly required by
those functions has been preserved.

Dependencies: torch, numpy, pandas, scikit-learn, and (optionally)
lightgbm for the LightGBM model definition.
"""
import wandb

import time, random
from collections import Counter
from typing import Any, Dict
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, HalvingRandomSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# justo después de los otros imports de sklearn


import lightgbm as lgb  # opcional

# ─────────────────────────────── Globals ──────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAR_TH = 1e-3  # Umbral de varianza mínima para conservar una dimensión latente

log = lambda *a, **k: print(time.strftime("%Y-%m-%d %H:%M:%S"), "|", *a, **k)

def variance_filter(X_train: np.ndarray, X_other: np.ndarray, threshold: float = VAR_TH) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.var(X_train, axis=0) > threshold
    return X_train[:, mask], X_other[:, mask]


# ──────────────────────── Clinical one‑hot ────────────────────────────

def clinical_onehot(subject_ids: np.ndarray) -> np.ndarray:
    """Devuelve one‑hot del sexo extraído de los IDs «..._M» / «..._F»."""
    sex = [s.split("_")[1] if "_" in s else "U" for s in subject_ids]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    return enc.fit_transform(np.array(sex).reshape(-1, 1))

# ─────────────────────── Latent extraction ────────────────────────────

class Encoder(torch.nn.Module):  # stub mínima – el usuario carga pesos aparte
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

def extract_latents(
    encoder: torch.nn.Module,
    X: torch.Tensor,
    *,
    with_sigma: bool = False,
    bs: int = 128,
) -> np.ndarray:
    """Codifica un tensor (N,C,H,W) → latentes. Con σ opcional."""
    loader = DataLoader(TensorDataset(X), bs, shuffle=False)
    encoder.eval().to(DEVICE)
    feats = []
    with torch.no_grad():
        for (xb,) in loader:
            mu, logv = encoder(xb.to(DEVICE))
            z = torch.cat([mu, torch.exp(0.5 * logv)], 1) if with_sigma else mu
            feats.append(z.cpu().numpy())
    Z = np.vstack(feats)
    log("Latentes:", Z.shape)
    return Z

# ───────────────────── Nested CV & models ─────────────────────────────

def _get_proba_or_score(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    score = clf.decision_function(X).reshape(-1, 1)
    return MinMaxScaler().fit_transform(score).ravel()

def _optimise_thr(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return thr[np.argmax(tpr - fpr)]

def _model_defs() -> Dict[str, tuple[Any, Dict[str, Any]]]:
    """Modelos ligeros + espacio de hiper‑parámetros mínimo."""
    return {
        "LogReg": (
            LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=SEED),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "LinearSVM": (
            LinearSVC(class_weight="balanced", dual=False, max_iter=4000, random_state=SEED),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "RandForest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 10]},
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                n_estimators=600,
                random_state=SEED,
                verbosity=-1,
                eval_metric="binary_logloss",
            ),
            {"clf__learning_rate": np.logspace(-2, -0.7, 6)},
        ),
    }

# ───────────────────────── run_cv mejorado ──────────────────────────
def run_cv(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    clinical: np.ndarray | None = None,
    tag: str = "mu",
    n_outer: int = 5,
    n_inner: int = 4,
) -> pd.DataFrame:
    """
    Nested-CV 5×4  +  HalvingRandomSearchCV  +  métricas extra  +  logging opcional.
    Devuelve un DataFrame con AUC, BAL y F1 por modelo/fold.
    """
    outer = StratifiedKFold(n_outer, shuffle=True, random_state=SEED)
    rows: list[dict[str, Any]] = []

    for k, (tr, te) in enumerate(outer.split(Z, y), 1):
        Ztr, Zte, ytr, yte = Z[tr], Z[te], y[tr], y[te]
        # Opcional: filtrado por varianza
        Z_tr, Z_va = variance_filter(Z_tr, Z_va)


        if clinical is not None:
            Ztr = np.hstack([Ztr, clinical[tr]])
            Zte = np.hstack([Zte, clinical[te]])

        for mdl, (base, space) in _model_defs().items():
            for use_pca in (False, True):
                # ---------- pipeline base ----------
                steps: list[tuple[str, Any]] = [("sc", StandardScaler())]
                if use_pca:
                    steps.append(("pca", PCA(n_components=0.95, random_state=SEED)))
                steps.append(("clf", base))
                pipe = Pipeline(steps)

                # ---------- búsqueda ----------
                inner = StratifiedKFold(n_inner, shuffle=True, random_state=SEED)
                search = HalvingRandomSearchCV(
                    pipe,
                    space,
                    factor=3,
                    scoring="balanced_accuracy",
                    cv=inner,
                    random_state=SEED,
                    n_jobs=-1,
                    verbose=0,
                )
                search.fit(Ztr, ytr)  # ← sin eval_set para evitar leakage


                # ---------- LightGBM calibrado ----------
                if mdl == "LightGBM":
                    Xt = best[:-1].transform(Ztr)
                    raw_clf = best[-1].set_params(early_stopping_rounds=None, eval_metric=None)
                    raw_clf = CalibratedClassifierCV(raw_clf, method="isotonic", cv=inner)
                    raw_clf.fit(Xt, ytr)
                    best = make_pipeline(*best[:-1], raw_clf)

                # ---------- predicciones ----------
                yprob = _get_proba_or_score(best, Zte)
                thr   = _optimise_thr(ytr, _get_proba_or_score(best, Ztr))
                yhat  = (yprob >= thr).astype(int)

                # ---------- métricas ----------
                bal = balanced_accuracy_score(yte, yhat)
                auc = roc_auc_score(yte, yprob)
                f1  = f1_score(yte, yhat)

                log(f"Fold {k} – {mdl:<10} | {'PCA' if use_pca else 'raw':3} | "
                    f"BAL={bal:.3f} AUC={auc:.3f} F1={f1:.3f}")

                # ---------- guardar resultados ----------
                rows.append({
                    "fold":  k,
                    "model": mdl,
                    "pca":   use_pca,
                    "bal":   bal,
                    "auc":   auc,
                    "f1":    f1,
                    "tag":   tag,
                })

                # ---------- logging a W&B (si existe) ----------
                if "wandb" in sys.modules:
                    wandb.log({f"CV_{tag}_{mdl}_{'PCA' if use_pca else 'raw'}_AUC": auc})

    return pd.DataFrame(rows)

