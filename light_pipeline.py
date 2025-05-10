#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight pipeline utilities for machine learning workflows, focusing on
Cross-Validation (CV), latent feature extraction from PyTorch encoders,
and basic data preprocessing steps.
"""

from __future__ import annotations

# Standard library imports
import os
import sys
import time
import random
import warnings
from collections import Counter
from typing import Any, Dict, Tuple, List # Added List for type hinting

# Third-party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn imports
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 # Enables HalvingRandomSearchCV
from sklearn.calibration import CalibratedClassifierCV # Not used in the provided snippet but kept if intended
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

# Weights & Biases import
import wandb

# ───────────────── Warning Filters ────────────────────────────
# Apply specific warning filters early.
warnings.filterwarnings(
    "ignore",
    message=r".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
    module=r"sklearn.*",
)

# ───────────────── Optional Project Settings Import ───────────────────
# Attempt to import project-specific settings; use defaults if not found.
try:
    from settings import VAR_TH as PROJECT_VAR_TH, NUM_WORKERS_T4 as PROJECT_NUM_WORKERS
    SETTINGS_AVAILABLE = True
    # Use project settings if available
    VAR_TH_DEFAULT = PROJECT_VAR_TH
    NUM_WORKERS_DEFAULT = PROJECT_NUM_WORKERS
    print("Successfully imported VAR_TH and NUM_WORKERS_T4 from settings.py.")
except ImportError:
    SETTINGS_AVAILABLE = False
    # Fallback default values if settings.py is not found or variables are not defined
    VAR_TH_DEFAULT = 1e-3
    NUM_WORKERS_DEFAULT = 0
    print("Warning: settings.py not found or VAR_TH/NUM_WORKERS_T4 not defined. Using fallback default values.")

# ───────────────── Global Constants and Configuration ───────────────────
SEED: int = 42  # Global seed for reproducibility

# --- Seed for Reproducibility ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True # Can impact performance
    # torch.backends.cudnn.benchmark = False   # Can impact performance

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for HalvingRandomSearchCV's min_resources calculation
MIN_RESOURCES_FRACTION: float = 0.2  # Fraction of training samples for min_resources
MIN_RESOURCES_ABSOLUTE: int = 20     # Absolute minimum number of samples for min_resources

# --- Global Variables (to be potentially configured by main script) ---
# These can be overridden by the main script if needed, e.g., from wandb.config
VAR_TH: float = VAR_TH_DEFAULT       # Variance threshold for feature filtering
NUM_WORKERS: int = NUM_WORKERS_DEFAULT # Number of workers for DataLoader

# --- Logging Utility ---
def log_message(message: str, *args, **kwargs) -> None:
    """Simple utility for timestamped print statements."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {message}", *args, **kwargs)

# ───────────────── Helper Functions ─────────────────────────────

def _min_resources(num_train_samples: int) -> int:
    """
    Calculates the `min_resources` parameter for HalvingRandomSearchCV
    based on the number of training samples.

    Args:
        num_train_samples (int): The number of samples in the training set.

    Returns:
        int: The calculated minimum resources value.
    """
    return max(MIN_RESOURCES_ABSOLUTE, int(MIN_RESOURCES_FRACTION * num_train_samples))


def filter_features_by_variance(
    X_train: np.ndarray,
    X_other: np.ndarray,
    threshold: float = VAR_TH  # Uses the globally defined VAR_TH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters features in X_train and X_other based on variance in X_train.
    Features in X_train with variance below the threshold are removed.
    The same feature mask is applied to X_other.

    Args:
        X_train (np.ndarray): Training data features.
        X_other (np.ndarray): Other data features (e.g., validation or test)
                              to be filtered with the same mask.
        threshold (float): The variance threshold. Features below this are removed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered X_train and X_other.
    """
    if X_train.shape[1] == 0: # No features to filter
        return X_train, X_other
        
    variances = np.var(X_train, axis=0)
    keep_mask = variances > threshold
    
    num_original_features = X_train.shape[1]
    num_kept_features = np.sum(keep_mask)
    
    if num_kept_features == 0:
        log_message(f"Warning: Variance filter (threshold={threshold:.1e}) removed all {num_original_features} features. "
                    "Returning original arrays to avoid downstream errors with empty feature sets.")
        return X_train, X_other # Or raise an error, or return empty arrays with original feature dim 0

    log_message(f"  VarianceFilter: Kept {num_kept_features}/{num_original_features} features (threshold={threshold:.1e}).")
    return X_train[:, keep_mask], X_other[:, keep_mask]


# ─────────────────── Clinical Data One-Hot Encoding ───────────────────

def clinical_onehot_encode_sex(subject_ids: np.ndarray) -> np.ndarray:
    """
    Extracts sex information from subject IDs and performs one-hot encoding.
    Assumes subject IDs might contain '_M' for male or '_F' for female.
    IDs without this pattern are categorized as 'Unknown' ('U').

    Args:
        subject_ids (np.ndarray): Array of subject ID strings.

    Returns:
        np.ndarray: One-hot encoded sex features.
    """
    sex_categories: List[str] = []
    for s_id in subject_ids:
        parts = s_id.split("_")
        if len(parts) > 1 and parts[1] in ("M", "F"):
            sex_categories.append(parts[1])
        else:
            sex_categories.append("U") # Unknown or unspecified

    encoder = OneHotEncoder(
        handle_unknown="ignore", # If new categories appear in test, they get all zeros
        sparse_output=False,     # Return dense array
        dtype=np.float32
    )
    # Reshape is necessary as OneHotEncoder expects a 2D array
    return encoder.fit_transform(np.array(sex_categories).reshape(-1, 1))


# ─────────────────────── Latent Feature Extraction ────────────────────────────

class EncoderStub(torch.nn.Module):
    """
    A minimal stub for an encoder.
    The actual encoder architecture and pre-trained weights are expected
    to be loaded and passed to `extract_latent_features`.
    This stub is primarily for type hinting and conceptual structure.
    """
    def __init__(self):
        super().__init__()
        # Actual layers would be defined here in a real encoder.

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Placeholder forward pass. A real encoder would implement its logic here.
        Should return mu and logvar for a VAE-like structure.
        """
        raise NotImplementedError(
            "This is an EncoderStub. A concrete Encoder implementation is required."
        )


def extract_latent_features(
    encoder_model: torch.nn.Module, # Should be a trained Encoder instance
    X_data: torch.Tensor,
    *,
    include_sigma_features: bool = False,
    batch_size: int = 128,
    device_to_use: torch.device = DEVICE # Use globally defined DEVICE
) -> np.ndarray:
    """
    Encodes input data (N, C, H, W) into latent features using the provided encoder.
    Optionally concatenates sigma (derived from logvar) to mu features.

    Args:
        encoder_model (torch.nn.Module): The trained PyTorch encoder model.
                                         It's assumed to have a forward pass
                                         that returns (mu, logvar).
        X_data (torch.Tensor): Input data tensor of shape (N, C, H, W).
        include_sigma_features (bool): If True, concatenates standard deviation
                                       (exp(0.5 * logvar)) to the mean (mu) features.
        batch_size (int): Batch size for processing through the encoder.
        device_to_use (torch.device): The device (CPU/GPU) to run encoding on.

    Returns:
        np.ndarray: A 2D numpy array of latent features (N, latent_dim) or
                    (N, 2 * latent_dim) if include_sigma_features is True.
    """
    dataset = TensorDataset(X_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    encoder_model.eval().to(device_to_use) # Set to eval mode and move to specified device
    
    all_latent_features: List[np.ndarray] = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for (x_batch,) in loader:
            x_batch = x_batch.to(device_to_use)
            mu, logvar = encoder_model(x_batch) # Assumes encoder returns mu and logvar

            if include_sigma_features:
                sigma = torch.exp(0.5 * logvar) # Calculate standard deviation
                current_batch_features = torch.cat([mu, sigma], dim=1)
            else:
                current_batch_features = mu
            
            all_latent_features.append(current_batch_features.cpu().numpy())

    Z_latents = np.vstack(all_latent_features)
    log_message(f"Extracted latent features shape: {Z_latents.shape} (sigma_included={include_sigma_features})")
    return Z_latents


# ───────────────────── Nested Cross-Validation & Model Utilities ────────────────────

def _get_probabilities_or_scores(classifier: Any, X_features: np.ndarray) -> np.ndarray:
    """
    Gets prediction probabilities (if available) or decision function scores
    from a trained classifier. Scores are scaled to [0, 1].

    Args:
        classifier (Any): Trained scikit-learn compatible classifier.
        X_features (np.ndarray): Features to predict on.

    Returns:
        np.ndarray: 1D array of probabilities or scaled scores.
    """
    if hasattr(classifier, "predict_proba"):
        # Return probabilities for the positive class
        return classifier.predict_proba(X_features)[:, 1]
    else:
        # Use decision_function and scale its output to [0, 1]
        # This is useful for models like SVM without direct probability output by default.
        scores = classifier.decision_function(X_features)
        if scores.ndim == 1: # If 1D, reshape for MinMaxScaler
            scores = scores.reshape(-1, 1)
        return MinMaxScaler().fit_transform(scores).ravel()


def _optimize_threshold_on_train(y_true_train: np.ndarray, y_pred_proba_train: np.ndarray) -> float:
    """
    Optimizes the classification threshold on training predictions
    using Youden's J statistic (maximizes TPR - FPR).

    Args:
        y_true_train (np.ndarray): True labels of the training set.
        y_pred_proba_train (np.ndarray): Predicted probabilities for the positive
                                         class on the training set.

    Returns:
        float: The optimal classification threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true_train, y_pred_proba_train)
    optimal_idx = np.argmax(tpr - fpr) # Youden's J statistic
    return thresholds[optimal_idx]


def _get_default_model_definitions() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Returns a dictionary of default scikit-learn models and their
    hyperparameter search spaces for use in `run_cross_validation`.

    Key format for hyperparameter space:
    - For direct estimator parameters: "param_name"
    - For Pipeline steps: "stepname__param_name" (e.g., "clf__C")

    Returns:
        Dict[str, Tuple[Any, Dict[str, Any]]]:
            A dictionary where keys are model names and values are tuples
            containing (estimator_instance, hyperparameter_search_space).
    """
    return {
        "LogReg": (
            LogisticRegression(
                solver="liblinear", # Good for L1/L2, smaller datasets
                class_weight="balanced",
                max_iter=5000, # Increased for convergence
                random_state=SEED,
            ),
            # IMPORTANT: For LogReg directly (not in a pipeline named 'clf'),
            # use param names directly like "C".
            # If LogReg is a step named 'clf' in a Pipeline, use "clf__C".
            {"C": [0.001, 0.01, 0.1, 1, 10, 100]}, # Parameter for LogisticRegression
        ),
        "LinearSVM": (
            # LinearSVC is often used in pipelines.
            # If used in a pipeline step named 'clf', params are 'clf__C'.
            LinearSVC(
                class_weight="balanced",
                dual="auto", # Changed from False to "auto" for broader compatibility
                max_iter=5000, # Increased for convergence
                random_state=SEED,
            ),
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}, # Assumes 'clf' is the step name in Pipeline
        ),
        "RandForest": (
            RandomForestClassifier(
                class_weight="balanced",
                n_jobs=-1, # Use all available cores
                random_state=SEED,
            ),
            # Assumes 'clf' is the step name in Pipeline
            {"clf__n_estimators": [50, 100, 200, 300],
             "clf__max_depth":   [None, 5, 10, 15, 20]},
        ),
    }


# ───────────────────────── Main Cross-Validation Function ──────────────────────────
def run_cross_validation(
    Z_features: np.ndarray,
    y_labels: np.ndarray,
    *,
    clinical_features: Optional[np.ndarray] = None,
    cv_run_tag: str = "mu_features", # Tag for logging (e.g., "mu", "mu_sigma")
    num_outer_folds: int = 5, # Original script used 3, common practice is 5 or 10
    num_inner_folds: int = 3, # Original script used 2
    use_variance_filter: bool = True, # Whether to apply variance filter
    variance_filter_thresh: float = VAR_TH # Threshold for variance filter
) -> pd.DataFrame:
    """
    Performs nested cross-validation with HalvingRandomSearchCV for hyperparameter tuning.
    Evaluates models on metrics like AUC, Balanced Accuracy, and F1-score.
    Logs results to W&B if available.

    Args:
        Z_features (np.ndarray): The input features for classification.
        y_labels (np.ndarray): The target labels.
        clinical_features (Optional[np.ndarray]): Optional clinical features to be
                                                 horizontally stacked with Z_features.
        cv_run_tag (str): A string tag to identify this CV run in logs (e.g., "mu_features").
        num_outer_folds (int): Number of folds for the outer CV loop (model evaluation).
        num_inner_folds (int): Number of folds for the inner CV loop (hyperparameter tuning).
        use_variance_filter (bool): If True, apply variance-based feature filtering.
        variance_filter_thresh (float): Threshold for variance filtering.

    Returns:
        pd.DataFrame: A DataFrame containing detailed results (AUC, BAL, F1)
                      per model, per fold, and configuration (e.g., with/without PCA).
    """
    outer_cv_strategy = StratifiedKFold(n_splits=num_outer_folds, shuffle=True, random_state=SEED)
    cv_results_rows: List[Dict[str, Any]] = [] # Store results for each fold/model

    model_definitions = _get_default_model_definitions()

    for fold_num, (train_indices, test_indices) in enumerate(outer_cv_strategy.split(Z_features, y_labels), 1):
        Z_train, Z_test = Z_features[train_indices], Z_features[test_indices]
        y_train, y_test = y_labels[train_indices], y_labels[test_indices]

        # --- Pre-fold diagnostics: Class distribution ---
        # (np.bincount with minlength=2 for binary classification CN=0, AD=1)
        train_class_counts = np.bincount(y_train, minlength=2)
        test_class_counts = np.bincount(y_test, minlength=2)

        log_message(f"Outer Fold {fold_num}/{num_outer_folds} – Class Distribution: "
                    f"Train CN={train_class_counts[0]}, AD={train_class_counts[1]} | "
                    f"Test CN={test_class_counts[0]}, AD={test_class_counts[1]}")

        # --- Skip fold if any partition (train/test) lacks both classes ---
        if not (all(train_class_counts > 0) and all(test_class_counts > 0)):
            log_message(f"⚠️ Fold {fold_num} skipped: One partition (train or test) does not contain both classes.")
            continue

        # --- Optional: Variance-based feature filtering ---
        if use_variance_filter:
            Z_train, Z_test = filter_features_by_variance(Z_train, Z_test, threshold=variance_filter_thresh)
            if Z_train.shape[1] == 0: # All features were filtered out
                log_message(f"⚠️ Fold {fold_num} skipped: All features removed by variance filter for this fold.")
                continue


        # --- Optional: Concatenate clinical features ---
        current_Z_train, current_Z_test = Z_train.copy(), Z_test.copy()
        if clinical_features is not None:
            clinical_train = clinical_features[train_indices]
            clinical_test = clinical_features[test_indices]
            current_Z_train = np.hstack([current_Z_train, clinical_train])
            current_Z_test = np.hstack([current_Z_test, clinical_test])
            log_message(f"  Fold {fold_num}: Concatenated clinical features. New Z_train shape: {current_Z_train.shape}")


        for model_name, (base_estimator, param_search_space) in model_definitions.items():
            # Iterate through configurations (e.g., with and without PCA)
            for apply_pca in (False, True):
                if current_Z_train.shape[1] == 0: # Should be caught by variance filter check earlier
                    log_message(f"  Fold {fold_num} - Model {model_name} ({'PCA' if apply_pca else 'Raw'}): Skipped, no features.")
                    continue

                # --- Define a scikit-learn Pipeline ---
                pipeline_steps: List[Tuple[str, Any]] = [("scaler", StandardScaler())]
                if apply_pca:
                    # Ensure n_components for PCA is valid
                    n_components_pca = min(0.95, current_Z_train.shape[0]-1, current_Z_train.shape[1]-1)
                    if isinstance(n_components_pca, float) or n_components_pca > 1 : # n_components must be < n_samples and < n_features if int
                         pipeline_steps.append(("pca", PCA(n_components=n_components_pca if n_components_pca > 1 else 0.95, random_state=SEED)))
                    else:
                        log_message(f"  Fold {fold_num} - Model {model_name} with PCA: Skipped, not enough samples/features for PCA ({current_Z_train.shape}).")
                        continue


                pipeline_steps.append(("clf", base_estimator)) # 'clf' is the name of the classifier step
                pipeline = Pipeline(steps=pipeline_steps)

                # --- Inner CV for Hyperparameter Search (HalvingRandomSearchCV) ---
                inner_cv_strategy = StratifiedKFold(n_splits=num_inner_folds, shuffle=True, random_state=SEED)
                
                # Adjust parameter search space keys if pipeline step name is 'clf'
                # Example: if original space was {"C": ...}, it becomes {"clf__C": ...}
                current_param_search_space = {f"clf__{k}" if not k.startswith("clf__") else k: v
                                               for k, v in param_search_space.items()}
                if model_name == "LogReg" and not apply_pca and "scaler" not in [s[0] for s in pipeline_steps]:
                    # If LogReg is used directly without a pipeline or PCA, params are direct
                    current_param_search_space = param_search_space


                search_cv = HalvingRandomSearchCV(
                    estimator=pipeline,
                    param_distributions=current_param_search_space,
                    factor=3, # Aggressiveness of halving
                    scoring="balanced_accuracy", # Or "roc_auc"
                    cv=inner_cv_strategy,
                    random_state=SEED,
                    n_jobs=-1, # Use all available cores
                    verbose=0, # Set to 1 or higher for more verbosity
                    min_resources=_min_resources(len(y_train)), # Dynamically set based on y_train size
                    error_score='raise' # Fail loudly if a combination errors out
                )

                try:
                    search_cv.fit(current_Z_train, y_train)
                except ValueError as ve:
                    log_message(f"⚠️ Fold {fold_num} - Model {model_name} ({'PCA' if apply_pca else 'Raw'}): "
                                f"HalvingRandomSearchCV failed. Error: {ve}. Skipping this config.")
                    continue
                
                best_model_pipeline = search_cv.best_estimator_

                # --- Predictions and Threshold Optimization ---
                # Get probabilities/scores on test set
                y_pred_proba_test = _get_probabilities_or_scores(best_model_pipeline, current_Z_test)
                
                # Optimize threshold on *training* data from this outer fold
                y_pred_proba_train_for_thresh = _get_probabilities_or_scores(best_model_pipeline, current_Z_train)
                optimal_threshold = _optimize_threshold_on_train(y_train, y_pred_proba_train_for_thresh)
                
                y_pred_binary_test = (y_pred_proba_test >= optimal_threshold).astype(int)

                # --- Calculate Metrics ---
                bal_acc = balanced_accuracy_score(y_test, y_pred_binary_test)
                auc = roc_auc_score(y_test, y_pred_proba_test)
                f1 = f1_score(y_test, y_pred_binary_test, zero_division=0) # Handles cases with no positive predictions

                config_name = f"{model_name}_{'PCA' if apply_pca else 'Raw'}"
                log_message(f"  Fold {fold_num} – {config_name:<18} | "
                            f"BAL_ACC={bal_acc:.4f} AUC={auc:.4f} F1={f1:.4f} (Thresh={optimal_threshold:.3f})")

                # --- Store Results ---
                cv_results_rows.append({
                    "fold": fold_num,
                    "model_config": config_name, # e.g., "LogReg_PCA" or "RandForest_Raw"
                    "model_base": model_name, # e.g., "LogReg"
                    "pca_applied": apply_pca,
                    "balanced_accuracy": bal_acc,
                    "auc": auc,
                    "f1_score": f1,
                    "cv_tag": cv_run_tag, # Tag for the overall CV run
                    "best_params": str(search_cv.best_params_) # Log best params as string
                })

                # --- Log to W&B (if active) ---
                if wandb.run is not None:
                    wandb.log({
                        f"CV_{cv_run_tag}_{config_name}_Fold{fold_num}_AUC": auc,
                        f"CV_{cv_run_tag}_{config_name}_Fold{fold_num}_BAL_ACC": bal_acc,
                        f"CV_{cv_run_tag}_{config_name}_Fold{fold_num}_F1": f1,
                    })

    # --- Aggregate and Log Full CV Results ---
    if not cv_results_rows:
        log_message(f"‼️ No results generated for CV run with tag '{cv_run_tag}'. Returning empty DataFrame.")
        return pd.DataFrame()

    all_cv_results_df = pd.DataFrame(cv_results_rows)

    if wandb.run is not None:
        try:
            # Log the entire results table to W&B
            wandb_results_table = wandb.Table(dataframe=all_cv_results_df)
            wandb.log({f"CV_Results_Table_{cv_run_tag}": wandb_results_table})

            # Log mean metrics per model_config to W&B summary
            mean_metrics_per_config = all_cv_results_df.groupby("model_config")[["auc", "balanced_accuracy", "f1_score"]].mean()
            for config_name, metrics_row in mean_metrics_per_config.iterrows():
                wandb.summary[f"CV_{cv_run_tag}_{config_name}_AUC_Mean"] = metrics_row.auc
                wandb.summary[f"CV_{cv_run_tag}_{config_name}_BAL_ACC_Mean"] = metrics_row.balanced_accuracy
                wandb.summary[f"CV_{cv_run_tag}_{config_name}_F1_Mean"] = metrics_row.f1_score
            log_message(f"Logged aggregated CV results for tag '{cv_run_tag}' to W&B.")

        except Exception as e:
            log_message(f"‼️ Error logging full CV results table or summary to W&B: {e}")
    
    # Check if the primary metric (LogReg AUC on mu-features) was computed
    # This is relevant if the main script expects "CV_mu_LogReg_AUC_mean"
    if cv_run_tag == "mu" or "mu_features" in cv_run_tag: # Check common tags for mu-only features
        logreg_raw_results = all_cv_results_df[all_cv_results_df["model_config"] == "LogReg_Raw"]
        if logreg_raw_results.empty:
            log_message("‼️ run_cross_validation: LogReg_Raw results not found for CV tag "
                        f"'{cv_run_tag}'. The primary W&B sweep metric might be missing or incorrect.")
        elif wandb.run is not None: # Ensure LogReg_Raw mean AUC is in summary if it's the target
            mean_logreg_raw_auc = logreg_raw_results["auc"].mean()
            wandb.summary[f"CV_{cv_run_tag}_LogReg_Raw_AUC_Mean"] = mean_logreg_raw_auc # Explicitly ensure it's there

    return all_cv_results_df

# Example usage (illustrative, typically called from another script)
if __name__ == "__main__":
    log_message("light_pipeline.py executed directly (for illustration/testing).")

    # --- Illustrative: Create dummy data ---
    num_samples = 100
    num_features_z = 50
    num_features_clinical = 3

    dummy_Z = np.random.rand(num_samples, num_features_z)
    dummy_y = np.random.randint(0, 2, num_samples)
    dummy_clinical = np.random.rand(num_samples, num_features_clinical)
    
    # --- Illustrative: Test clinical_onehot_encode_sex ---
    dummy_ids = np.array([f"Subj{i:03d}_{random.choice(['M', 'F', 'X'])}" for i in range(num_samples)])
    sex_onehot = clinical_onehot_encode_sex(dummy_ids)
    log_message(f"Dummy sex one-hot encoded shape: {sex_onehot.shape}")

    # --- Illustrative: Test run_cross_validation ---
    # Initialize W&B for testing if you want to see logs
    # try:
    #     wandb.init(project="light_pipeline_test", name="cv_test_run", mode="disabled") # mode="disabled" or "online"
    # except Exception as e:
    #     log_message(f"W&B init for testing failed: {e}")


    log_message("\nRunning illustrative cross-validation with dummy data...")
    cv_results = run_cross_validation(
        dummy_Z,
        dummy_y,
        clinical_features=sex_onehot, # Use the generated sex features
        cv_run_tag="dummy_data_test",
        num_outer_folds=2, # Fewer folds for quick test
        num_inner_folds=2
    )

    if not cv_results.empty:
        log_message("\n--- Illustrative CV Results (Mean per Model Config) ---")
        mean_results = cv_results.groupby("model_config")[["auc", "balanced_accuracy", "f1_score"]].mean()
        print(mean_results)
    else:
        log_message("Illustrative CV run produced no results.")

    # if wandb.run is not None:
    #     wandb.finish()


