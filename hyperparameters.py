#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep-ready training and evaluation script for a β-VAE model followed by
a classifier, optimized for Weights & Biases (W&B) sweeps targeting AUC
from post-training Cross-Validation.
"""

from __future__ import annotations

# Standard library imports
import os
import sys
import random
import math
import time
from collections import Counter
from typing import Dict, Tuple, Any, Optional, Sequence

# Third-party imports
import numpy as np
import pandas as pd # Added import, as it's used in _log_df_means
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.backends.cuda as torch_back
# amp is already imported above
#from torch import amp
import os
os.environ["TORCH_INDUCTOR_DISABLE_MAX_AUTOTUNE_GEMM"] = "1"


# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv # noqa: F401, E402 # Keep for HalvingRandomSearchCV
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Weights & Biases import
import wandb

# --- Warning Handling ---
# Apply warning filters early, before other sklearn imports if possible.
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
    module=r"sklearn.*",
)
warnings.filterwarnings('ignore', category=UserWarning) # General UserWarning suppression as in original main()

# --- W&B Service Requirement ---
# os.environ["WANDB_REQUIRE_SERVICE"] = "True" # Force modern service if needed
#wandb.require("service")

# --- Project-specific imports & Path Configuration ---
# Assuming 'settings.py' and 'light_pipeline.py' are in the same directory or accessible via PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Add script directory to sys.path to allow direct import of local modules
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    print(f"Added {script_dir} to sys.path")

# Conditional import for light_pipeline
LIGHT_PIPELINE_AVAILABLE = False
try:
    from light_pipeline import extract_latent_features as lp_extract_latent_features, \
                               clinical_onehot_encode_sex as lp_clinical_onehot_encode_sex, \
                               run_cross_validation as lp_run_cross_validation, \
                               _min_resources as lp_min_resources
    # Use aliased names to avoid potential conflicts if these names are defined elsewhere
    LIGHT_PIPELINE_AVAILABLE = True
    print("Successfully imported from light_pipeline.py")
except ImportError as e:
    print(f"Warning: Could not import from light_pipeline.py. Error: {e}")
    print(f"Current sys.path: {sys.path}")
    # Define dummy functions if light_pipeline is not available
    def lp_extract_latent_features(*args, **kwargs):
        raise NotImplementedError("light_pipeline.py not available, lp_extract_latent_features cannot be used.")
    def lp_clinical_onehot_encode_sex(*args, **kwargs):
        raise NotImplementedError("light_pipeline.py not available, lp_clinical_onehot_encode_sex cannot be used.")
    def lp_run_cross_validation(*args, **kwargs):
        raise NotImplementedError("light_pipeline.py not available, lp_run_cross_validation cannot be used.")
    def lp_min_resources(*args, **kwargs):
        raise NotImplementedError("light_pipeline.py not available, lp_min_resources cannot be used.")

# Import from settings after path modification (if settings.py is local)
try:
    from settings import VAR_TH_DEFAULT, NUM_WORKERS_T4_DEFAULT # Renamed to avoid conflict with global VAR_TH
    VAR_TH = VAR_TH_DEFAULT
    NUM_WORKERS_T4 = NUM_WORKERS_T4_DEFAULT
    print(f"Successfully imported VAR_TH ({VAR_TH}) and NUM_WORKERS_T4 ({NUM_WORKERS_T4}) from settings.py")
except ImportError:
    print("Warning: settings.py not found or VAR_TH/NUM_WORKERS_T4 not defined. Using fallback values.")
    VAR_TH = 1e-5 # Fallback value, ensure this is appropriate
    NUM_WORKERS_T4 = 0 # Fallback value

# --- PyTorch Accelerators ---
# Configure PyTorch for performance
torch_back.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True # Good for fixed input sizes
torch.set_float32_matmul_precision("high") # Or "medium"

# ────────────────────── Global Configuration & Constants ──────────────────────
PROJECT_DIR: str = "/content/drive/MyDrive/AAL_166" # Consider making this configurable
CURRENT_FOLD: int = 1 # Renamed from FOLD to avoid ambiguity
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED: int = 42

# --- Seed for Reproducibility ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Potentially add: torch.backends.cudnn.deterministic = True (can impact performance)

# --- W&B Sweep Default Hyperparameters ---
# These are used if not overridden by the W&B sweep configuration.
WANDB_DEFAULTS: Dict[str, Any] = dict(
    lr=1e-4,                # Learning rate for VAE optimizer
    batch_size=64,          # Batch size for VAE training
    beta=25,                # β factor for KLD term in VAE loss
    beta_ramp_epochs=400,   # Number of epochs to ramp up β to its full value
    epochs=450,             # Maximum number of VAE training epochs
    k_filters=32,           # Base number of filters in VAE convolutional layers
    latent_dim=128,         # Dimensionality of the VAE's latent space
    eval_interval=25,       # Epoch interval for performing quick evaluation metrics
    patience=150,           # Patience for early stopping based on validation loss
    # VAR_TH = 1e-3 # This was commented out, VAR_TH is now from settings.py or fallback
)

# --- VAE Training Constants ---
TOL_REL: float = 1e-3     # Relative tolerance for validation loss improvement (0.1%)
CLIP_NORM: float = 1.0    # Max L2 norm for gradient clipping
FREE_NATS: float = 5.0    # Minimum KLD value per latent dimension (Kingma et al., ICLR 2017)

# ────────────────────── Data Loading Utilities ─────────────────────────

def verify_ad_cn_balance(y_tensor: torch.Tensor, dataset_name: str) -> None:
    """
    Verifies that a dataset split contains at least one instance of AD (Alzheimer's Disease)
    and CN (Cognitively Normal) labels. Aborts if not.

    Args:
        y_tensor (torch.Tensor): Tensor of labels (0 for CN, 1 for AD, 2 for Other).
        dataset_name (str): Name of the dataset split (e.g., "Train", "Validation").
    
    Raises:
        RuntimeError: If the split does not contain both AD and CN classes.
    """
    y_numpy = y_tensor.numpy()
    # Filter out 'Other' class (label 2) for AD/CN balance check
    y_ad_cn_only = y_numpy[y_numpy != 2]
    unique_classes = np.unique(y_ad_cn_only)

    if len(unique_classes) < 2:
        error_message = (
            f"❌ ABORTED: The '{dataset_name}' split contains only one class among AD and CN. "
            f"Found classes: {unique_classes}. Both AD (1) and CN (0) are required."
        )
        print(error_message)
        raise RuntimeError(error_message)
    else:
        print(f"✅ '{dataset_name}' contains both AD and CN classes.")


def _load_torch_file(directory: str, filename: str) -> Any:
    """
    Helper function to load a .pt file with error handling.

    Args:
        directory (str): The directory where the file is located.
        filename (str): The name of the file to load.

    Returns:
        Any: The loaded data from the torch file.

    Raises:
        FileNotFoundError: If the file is not found.
        Exception: For other loading errors.
    """
    filepath = os.path.join(directory, filename)
    try:
        # weights_only=False is the default and allows loading arbitrary pickled objects.
        # If you only save tensors or model states, consider True for security.
        return torch.load(filepath, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"❌ Error loading file {filepath}: {e}")
        raise


def load_fold_data(fold_index: int = 1) -> Tuple[torch.Tensor, ...]:
    """
    Loads all data splits (train, validation, test) and their corresponding
    labels for a given fold. Labels are encoded as AD=1, CN=0, Other=2.

    Args:
        fold_index (int): The fold number to load data for.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing:
            trX (train features), vaX (validation features), teX (test features),
            trL (train original labels), vaL (validation original labels), teL (test original labels),
            trY (train encoded labels), vaY (validation encoded labels), teY (test encoded labels).
    """
    fold_dir = os.path.join(PROJECT_DIR, f"fold_{fold_index}")
    print(f"Loading data from directory: {fold_dir}")

    # Load features (assumed to be z-normalized)
    trX = _load_torch_file(fold_dir, "train_z.pt")
    vaX = _load_torch_file(fold_dir, "val_z.pt")
    teX = _load_torch_file(fold_dir, "test_z.pt")

    # Load original string labels
    trL_original = _load_torch_file(fold_dir, "train_labels.pt")
    vaL_original = _load_torch_file(fold_dir, "val_labels.pt")
    teL_original = _load_torch_file(fold_dir, "test_labels.pt")

    # Define label encoding function
    def encode_label(s: str) -> int:
        if isinstance(s, str):
            if s.startswith("AD_"): return 1  # Alzheimer's Disease
            if s.startswith("CN_"): return 0  # Cognitively Normal
        return 2  # Other categories

    # Encode labels
    trY = torch.tensor([encode_label(s) for s in trL_original], dtype=torch.long)
    vaY = torch.tensor([encode_label(s) for s in vaL_original], dtype=torch.long)
    teY = torch.tensor([encode_label(s) for s in teL_original], dtype=torch.long)

    print("\n--- Data Shapes ---")
    print(f"  Train X: {trX.shape}, Y: {trY.shape}")
    print(f"  Val   X: {vaX.shape}, Y: {vaY.shape}")
    print(f"  Test  X: {teX.shape}, Y: {teY.shape}")

    print("\n--- Label Counts (0:CN, 1:AD, 2:Other) ---")
    print(f"  Train: {Counter(trY.numpy())}")
    print(f"  Val:   {Counter(vaY.numpy())}")
    print(f"  Test:  {Counter(teY.numpy())}")

    # Verify AD/CN balance in each split
    print("\n--- Verifying AD/CN Balance (excluding 'Other') ---")
    verify_ad_cn_balance(trY, "Train")
    verify_ad_cn_balance(vaY, "Validation")
    verify_ad_cn_balance(teY, "Test")

    return trX, vaX, teX, trL_original, vaL_original, teL_original, trY, vaY, teY


# ────────────────────── β-VAE Architecture ─────────────────────────────

class Encoder(nn.Module):
    """
    Encoder part of the β-VAE. Maps input images to latent space parameters (μ and logσ²).
    Input images are assumed to be 166x166.
    """
    def __init__(self, input_channels: int, latent_dim: int, k_filters_base: int = 32):
        super().__init__()
        print(f"Initializing Encoder: input_channels={input_channels}, latent_dim={latent_dim}, k_filters_base={k_filters_base}")
        k = k_filters_base
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1), # Output: k x 82 x 82
            nn.Conv2d(k, 2*k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),          # Output: 2k x 39 x 39
            nn.Conv2d(2*k, 4*k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),        # Output: 4k x 18 x 18 (approx, check padding)
            nn.Conv2d(4*k, 8*k, kernel_size=3, stride=2, padding=0), nn.LeakyReLU(0.1),        # Output: 8k x 8 x 8 (approx, check padding)
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            # Assuming input image size is 166x166, as per dummy_input
            dummy_input = torch.zeros(1, input_channels, 166, 166)
            conv_output_shape = self.conv_layers(dummy_input).shape
            self.flattened_size = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]
            print(f"  Calculated flattened size after convolutions: {self.flattened_size} (Shape: {conv_output_shape})")

        self.flatten = nn.Flatten()
        # Intermediate FC layer size, e.g., one-tenth of flattened_size or a fixed reasonable value
        fc_intermediate_size = max(latent_dim * 4, self.flattened_size // 10) # Ensure it's reasonably large
        print(f"  FC intermediate size: {fc_intermediate_size}")

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, fc_intermediate_size),
            nn.LeakyReLU(0.1)
        )
        self.fc_mu = nn.Linear(fc_intermediate_size, latent_dim)
        self.fc_logvar = nn.Linear(fc_intermediate_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean (μ) and log variance (logσ²) of the latent distribution.
        """
        h_conv = self.conv_layers(x)
        h_flat = self.flatten(h_conv)
        h_fc = self.fc_layers(h_flat)
        mu = self.fc_mu(h_fc)
        logvar = self.fc_logvar(h_fc)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder part of the β-VAE. Maps latent variables back to image space.
    Designed to be roughly symmetrical to the Encoder.
    """
    def __init__(self, latent_dim: int, output_channels: int, k_filters_base: int = 32):
        super().__init__()
        print(f"Initializing Decoder: latent_dim={latent_dim}, output_channels={output_channels}, k_filters_base={k_filters_base}")
        k = k_filters_base
        self.k8_filters = 8 * k
        # These dimensions should correspond to the output of the Encoder's last conv layer before flatten
        # For example, if Encoder's last conv output is (8k, 9, 9)
        self.reshape_dims = (self.k8_filters, 9, 9) # Critical: Must match encoder output spatial dims
        fc_output_size = self.k8_filters * self.reshape_dims[1] * self.reshape_dims[2]
        print(f"  Decoder FC output size (to be reshaped): {fc_output_size}, Reshape dims: {self.reshape_dims}")

        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, fc_output_size),
            nn.LeakyReLU(0.1)
        )
        self.deconv_layers = nn.Sequential(
            # Input: 8k x 9 x 9 (adjust kernel, stride, padding to reconstruct)
            nn.ConvTranspose2d(self.k8_filters, 4*k, kernel_size=3, stride=2, padding=0), nn.LeakyReLU(0.1), # to ~18x18
            nn.ConvTranspose2d(4*k, 2*k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),             # to ~39x39
            nn.ConvTranspose2d(2*k, k, kernel_size=4, stride=2, padding=0), nn.LeakyReLU(0.1),                # to ~82x82
            nn.ConvTranspose2d(k, output_channels, kernel_size=4, stride=2, padding=0)                         # to ~166x166
            # Final layer might use Sigmoid or Tanh if inputs are normalized to [0,1] or [-1,1]
            # If MSE loss is used on raw pixel values, no activation is strictly needed here.
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Latent variable tensor (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        h_fc = self.fc_layers(z)
        h_reshaped = h_fc.view(-1, *self.reshape_dims) # Reshape to (batch_size, channels, height, width)
        x_reconstructed = self.deconv_layers(h_reshaped)
        return x_reconstructed


class BetaVAE(nn.Module):
    """
    β-Variational Autoencoder model.
    """
    def __init__(self, input_channels: int, latent_dim: int, k_filters_base: int):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim, k_filters_base)
        self.decoder = Decoder(latent_dim, input_channels, k_filters_base)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(μ, σ²) by sampling from N(0, I).
        σ = exp(0.5 * logσ²)
        z = μ + σ * ε, where ε ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std) # Sample from N(0, I)
        return mu + epsilon * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the β-VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_reconstructed: The reconstructed input.
                - mu: Mean of the latent distribution.
                - logvar: Log variance of the latent distribution.
        """
        mu, logvar = self.encoder(x)
        z = self._reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar


def vae_loss_function(
    x_reconstructed: torch.Tensor,
    x_original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    free_nats_clamp: float = FREE_NATS
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the β-VAE loss.
    Loss = Reconstruction Loss + β * KLD
    KLD is clamped at `free_nats_clamp` as per Kingma et al. (2017).

    Args:
        x_reconstructed (torch.Tensor): Reconstructed input data.
        x_original (torch.Tensor): Original input data.
        mu (torch.Tensor): Mean of the latent distribution.
        logvar (torch.Tensor): Log variance of the latent distribution.
        beta (float): Weight of the KLD term.
        free_nats_clamp (float): Minimum value for KLD before averaging (information bottleneck).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: The total β-VAE loss.
            - reconstruction_loss: The MSE reconstruction loss.
            - kld_loss: The Kullback-Leibler divergence.
    """
    # Reconstruction loss (Mean Squared Error, averaged over batch)
    recon_loss = F.mse_loss(x_reconstructed, x_original, reduction='sum') / x_original.size(0)

    # Kullback-Leibler Divergence (KLD)
    # KLD = -0.5 * Σ (1 + log(σ²) - μ² - σ²)
    # Sum over latent dimensions, then average over batch
    kld_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_per_sample = kld_elementwise.sum(dim=1) # Sum over latent dimensions

    # Apply free nats (information bottleneck)
    # This encourages the model to use at least 'free_nats_clamp' nats of information for each sample.
    clamped_kld = torch.clamp(kld_per_sample, min=free_nats_clamp)
    kld_loss = clamped_kld.mean() # Average over the batch

    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss


@torch.no_grad()
def encode_to_mu(encoder: Encoder, X_data: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """
    Encodes input data X_data to their latent means (μ) using the provided encoder.
    Runs in evaluation mode and without gradient tracking.

    Args:
        encoder (Encoder): The trained VAE encoder.
        X_data (torch.Tensor): Input data tensor.
        batch_size (int): Batch size for processing.

    Returns:
        np.ndarray: Array of latent means (μ).
    """
    encoder.eval() # Set encoder to evaluation mode
    all_mu = []
    dataset = TensorDataset(X_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    for (x_batch,) in loader:
        x_batch = x_batch.to(DEVICE, memory_format=torch.channels_last) # Optimize memory format for conv
        # Use Automatic Mixed Precision (AMP) if on CUDA for potentially faster inference
        with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
            mu, _ = encoder(x_batch)
        all_mu.append(mu.cpu().float().numpy()) # Move to CPU, convert to float32 numpy

    return np.vstack(all_mu)

# ────────────────────── VAE Training Function ───────────────────────────
def train_vae(
    config: wandb.Config,
    train_X: torch.Tensor, val_X: torch.Tensor,
    train_Y: torch.Tensor, val_Y: torch.Tensor
) -> Tuple[BetaVAE, Dict[str, Any]]:
    """
    Trains the Beta-VAE model.

    Args:
        config (wandb.Config): Configuration object from W&B, containing hyperparameters.
        train_X (torch.Tensor): Training features.
        val_X (torch.Tensor): Validation features for reconstruction loss.
        train_Y (torch.Tensor): Training labels (for intermediate AD/CN classification).
        val_Y (torch.Tensor): Validation labels (for intermediate AD/CN classification).

    Returns:
        Tuple[BetaVAE, Dict[str, Any]]:
            - trained_vae: The VAE model with the best weights loaded.
            - best_model_state_dict: State dictionary of the best performing model.
    """
    print("\n--- Starting VAE Training ---")
    print(f"Using device: {DEVICE}")
    print("Sweep Configuration (from wandb.config):")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # --- Data Preparation for Intermediate AD/CN Evaluation ---
    # These masks select only AD (1) and CN (0) samples, excluding 'Other' (2)
    mask_train_adcn = train_Y != 2
    mask_val_adcn = val_Y != 2

    train_X_adcn = train_X[mask_train_adcn]
    train_Y_adcn_numpy = train_Y[mask_train_adcn].numpy()
    val_X_adcn = val_X[mask_val_adcn]
    val_Y_adcn_numpy = val_Y[mask_val_adcn].numpy()

    print("\nData for Intermediate AD/CN Evaluation (Logistic Regression on μ):")
    print(f"  Train AD/CN X shape: {train_X_adcn.shape}, Y shape: {train_Y_adcn_numpy.shape}")
    print(f"  Val AD/CN X shape: {val_X_adcn.shape}, Y shape: {val_Y_adcn_numpy.shape}")

    # VAE training uses all available data (train_X in this setup, could be train_X + val_X)
    # The original script used trX.clone(), implying only training set for VAE.
    vae_training_data_X = train_X.clone()
    print(f"\nVAE Training Data shape (using only train_X): {vae_training_data_X.shape}")

    vae_train_dataset = TensorDataset(vae_training_data_X)
    train_loader = DataLoader(
        vae_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True, # Helps speed up CPU to GPU transfers
        num_workers=NUM_WORKERS_T4,
        persistent_workers=(NUM_WORKERS_T4 > 0), # Keep workers alive
        prefetch_factor=2 if NUM_WORKERS_T4 > 0 else None, # Number of batches to prefetch
        drop_last=True, # Drop last incomplete batch for consistent batch stats
    )

    # DataLoader for validation reconstruction loss (uses val_X)
    val_recon_dataset = TensorDataset(val_X)
    val_recon_loader = DataLoader(val_recon_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # --- Model, Optimizer, AMP Initialization ---
    input_channels = train_X.shape[1]
    vae_model = BetaVAE(input_channels, config.latent_dim, config.k_filters).to(DEVICE, memory_format=torch.channels_last)
    
    # Compile model for potential speedup (PyTorch 2.0+)
    # "reduce-overhead" is good for smaller models or when compilation time is a concern.
    # "max-autotune" might offer more speedup but takes longer to compile.
    print(f"Compiling VAE model with mode: 'reduce-overhead'")
    try:
        vae_model = torch.compile(vae_model, mode="reduce-overhead")
    except Exception as e:
        print(f"Model compilation failed: {e}. Proceeding without compilation.")


    wandb.watch(vae_model, log="gradients", log_freq=100) # Log gradients and model topology

    # AdamW optimizer with fused option for CUDA
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=config.lr, fused=(DEVICE.type == 'cuda'))

    # Automatic Mixed Precision (AMP) setup
    amp_enabled = DEVICE.type == 'cuda'
    # bfloat16 is generally preferred over float16 if supported, for better stability.
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    # GradScaler helps prevent underflow/overflow issues with float16
    amp_enabled = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_enabled) 
    #scaler = 	torch.amp.GradScaler(device_type="cuda")
    print(f"Automatic Mixed Precision (AMP) Enabled: {amp_enabled}, dtype: {amp_dtype}")

    # --- Training Loop ---
    best_val_total_loss = math.inf
    epochs_without_improvement = 0
    best_model_state_dict = None

    # Get training parameters from config or defaults
    eval_interval = getattr(config, "eval_interval", WANDB_DEFAULTS['eval_interval'])
    patience_epochs = getattr(config, "patience", WANDB_DEFAULTS['patience'])

    print(f"\nStarting training for {config.epochs} epochs...")
    training_start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        vae_model.train() # Set model to training mode
        
        current_total_loss, current_recon_loss, current_kld_loss = 0.0, 0.0, 0.0

        for i, (x_batch,) in enumerate(train_loader):
            x_batch = x_batch.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # More efficient than setting to zero

            with torch.amp.autocast(device_type="cuda",dtype=amp_dtype):
                x_reconstructed, mu, logvar = vae_model(x_batch)
                
                # Beta scheduling: ramp up beta using a square root curve
                # progress = current_epoch / total_ramp_epochs
                # beta_t = beta_max * sqrt(progress) for progress <= 1, else beta_max
                ramp_progress = epoch / max(config.beta_ramp_epochs, 1) # Avoid division by zero
                effective_beta = min(config.beta, config.beta * (ramp_progress**0.5))

                loss, recon, kld = vae_loss_function(x_reconstructed, x_batch, mu, logvar, effective_beta)

            scaler.scale(loss).backward() # Scale loss for AMP
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), CLIP_NORM) # Gradient clipping
            scaler.step(optimizer) # Optimizer step
            scaler.update() # Update scaler for next iteration

            current_total_loss += loss.item()
            current_recon_loss += recon.item()
            current_kld_loss += kld.item()

        # Average training losses for the epoch
        num_train_batches = len(train_loader)
        avg_epoch_train_loss = current_total_loss / num_train_batches
        avg_epoch_train_recon = current_recon_loss / num_train_batches
        avg_epoch_train_kld = current_kld_loss / num_train_batches

        # --- Validation Phase ---
        vae_model.eval() # Set model to evaluation mode
        val_epoch_total_loss, val_epoch_recon_loss, val_epoch_kld_loss = 0.0, 0.0, 0.0
        with torch.no_grad(): # Disable gradient calculations for validation
            for (x_val_batch,) in val_recon_loader:
                x_val_batch = x_val_batch.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
                with torch.amp.autocast(device_type="cuda",dtype=amp_dtype):
                    x_val_reconstructed, mu_val, logvar_val = vae_model(x_val_batch)
                    # Use the full beta for validation loss calculation
                    v_loss, v_recon, v_kld = vae_loss_function(
                        x_val_reconstructed, x_val_batch, mu_val, logvar_val, config.beta
                    )
                val_epoch_total_loss += v_loss.item()
                val_epoch_recon_loss += v_recon.item()
                val_epoch_kld_loss += v_kld.item()
        
        avg_epoch_val_total_loss = val_epoch_total_loss / len(val_recon_loader)
        avg_epoch_val_recon_loss = val_epoch_recon_loss / len(val_recon_loader)
        avg_epoch_val_kld_loss = val_epoch_kld_loss / len(val_recon_loader)
        
        epoch_duration_sec = time.time() - epoch_start_time

        # --- Logging to W&B ---
        log_payload = {
            "epoch": epoch,
            "train_loss_total": avg_epoch_train_loss,
            "train_loss_recon": avg_epoch_train_recon,
            "train_loss_kld": avg_epoch_train_kld,
            "val_loss_total": avg_epoch_val_total_loss,
            "val_loss_recon": avg_epoch_val_recon_loss,
            "val_loss_kld": avg_epoch_val_kld_loss,
            "beta_effective": effective_beta,
            "epoch_duration_sec": epoch_duration_sec,
        }
        if DEVICE.type == 'cuda':
            log_payload["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(DEVICE) / (1024**3)
            log_payload["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(DEVICE) / (1024**3)

        # Perform quick AD/CN classification metrics at specified intervals
        if eval_interval > 0 and (epoch % eval_interval == 0 or epoch == 1 or epoch == config.epochs):
            # Ensure encoder is passed correctly (vae_model.encoder)
            acc_tr, auc_tr, acc_va, auc_va = run_quick_metrics(
                vae_model.encoder, train_X_adcn, train_Y_adcn_numpy, val_X_adcn, val_Y_adcn_numpy
            )
            log_payload.update({
                "QuickMetrics_Train_ACC_mu": acc_tr,
                "QuickMetrics_Train_AUC_mu": auc_tr,
                "QuickMetrics_Val_ACC_mu": acc_va,
                "QuickMetrics_Val_AUC_mu": auc_va,
            })
            # Update W&B summary for easy tracking of best validation AUC
            wandb.summary["VAL_LogReg_AUC_Quick"] = auc_va 
            # The dict version seems specific, ensure it's needed for your dashboard
            wandb.summary["VAL_LogReg_AUC_Quick_Per_Fold"] = {f"fold{CURRENT_FOLD}": auc_va}


        try:
            wandb.log(log_payload)
        except Exception as e:
            print(f"[W&B LOGGING ERROR at epoch {epoch}] {e}")

        print(f"Epoch {epoch}/{config.epochs} [{epoch_duration_sec:.2f}s] - "
              f"Train Loss: {avg_epoch_train_loss:.3f} (Recon: {avg_epoch_train_recon:.3f}, KLD: {avg_epoch_train_kld:.3f}) | "
              f"Val Total Loss: {avg_epoch_val_total_loss:.3f} (Recon: {avg_epoch_val_recon_loss:.3f}) | Beta: {effective_beta:.2f}")

        # --- Early Stopping Check (based on validation total loss) ---
        if avg_epoch_val_total_loss < best_val_total_loss * (1 - TOL_REL):
            print(f"  Validation total loss improved ({best_val_total_loss:.4f} -> {avg_epoch_val_total_loss:.4f}). Saving model state.")
            best_val_total_loss = avg_epoch_val_total_loss
            epochs_without_improvement = 0
            # Save the state dict of the best model (on CPU to save GPU memory)
            best_model_state_dict = {k: v.detach().clone().cpu() for k, v in vae_model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            print(f"  Validation total loss did not improve for {epochs_without_improvement} epoch(s). Current best: {best_val_total_loss:.4f}")

        if epochs_without_improvement >= patience_epochs:
            print(f"\nEarly stopping triggered after {epoch} epochs due to no improvement in validation total loss for {patience_epochs} epochs.")
            break

    # --- End of Training ---
    total_training_duration_sec = time.time() - training_start_time
    print(f"\n--- VAE Training Finished ---")
    print(f"Total training time: {total_training_duration_sec:.2f} seconds ({total_training_duration_sec/60:.2f} minutes)")
    print(f"Stopped at epoch: {epoch}")

    if best_model_state_dict:
        print(f"Loading best model state (Validation Total Loss: {best_val_total_loss:.4f}) from epoch {epoch - epochs_without_improvement}")
        # Ensure the model is on the correct device before loading state_dict if it was moved
        vae_model.load_state_dict(best_model_state_dict)
    else:
        print("Warning: No improvement observed or early stopping patience was 0. Using model from the final epoch.")

    # Log final summary metrics to W&B
    wandb.summary["best_val_total_loss_VAE"] = best_val_total_loss
    wandb.summary["stopped_epoch_VAE"] = epoch
    wandb.summary["total_training_time_min_VAE"] = total_training_duration_sec / 60

    return vae_model, best_model_state_dict


# ─────────────────── Quick Metrics Evaluation (on μ-features) ────────────
def run_quick_metrics(
    encoder_model: Encoder,
    X_train_adcn: torch.Tensor, y_train_adcn: np.ndarray,
    X_val_adcn: torch.Tensor, y_val_adcn: np.ndarray,
    variance_threshold: float = VAR_TH # Use the global VAR_TH
) -> Tuple[float, float, float, float]:
    """
    Performs a quick evaluation using μ-features from the encoder.
    The process involves:
    1. Encoding data to μ-features.
    2. Standardizing (z-scoring) the features.
    3. Filtering out features with variance below `variance_threshold`.
    4. Training a Logistic Regression classifier (balanced, no extensive hyperparameter search).
    5. Calculating Accuracy and AUC on train and validation sets.

    Args:
        encoder_model (Encoder): The VAE's encoder module.
        X_train_adcn (torch.Tensor): Training features (AD/CN only).
        y_train_adcn (np.ndarray): Training labels (AD/CN only).
        X_val_adcn (torch.Tensor): Validation features (AD/CN only).
        y_val_adcn (np.ndarray): Validation labels (AD/CN only).
        variance_threshold (float): Minimum variance for a latent dimension to be kept.

    Returns:
        Tuple[float, float, float, float]:
            - acc_train: Accuracy on the training set.
            - auc_train: AUC on the training set.
            - acc_val: Accuracy on the validation set.
            - auc_val: AUC on the validation set.
    """
    eval_start_time = time.time()

    # 1. Encode data to μ-features using the provided encoder
    # Ensure encoder is on the correct device for encoding
    encoder_model.to(DEVICE)
    Z_train_mu = encode_to_mu(encoder_model, X_train_adcn)
    Z_val_mu = encode_to_mu(encoder_model, X_val_adcn)

    # 2. Standardize features (z-score)
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train_mu)
    Z_val_scaled = scaler.transform(Z_val_mu)

    # 3. Filter dimensions with very low variance (on training set)
    train_variances = np.var(Z_train_scaled, axis=0)
    feature_mask = train_variances > variance_threshold
    
    num_original_features = Z_train_scaled.shape[1]
    Z_train_filtered = Z_train_scaled[:, feature_mask]
    Z_val_filtered = Z_val_scaled[:, feature_mask]
    num_kept_features = Z_train_filtered.shape[1]
    
    print(f"  QuickMetrics: Kept {num_kept_features}/{num_original_features} latent features after variance threshold ({variance_threshold:.1e}).")
    if num_kept_features == 0:
        print("  QuickMetrics: Warning! No features kept after variance filtering. AUC/ACC will be 0.5/random.")
        # Return neutral scores if no features are left
        return 0.5, 0.5, 0.5, 0.5


    # 4. Train a simple Logistic Regression classifier
    # 'liblinear' is good for smaller datasets and L1/L2 penalties.
    # 'saga' could be an alternative for larger datasets.
    log_reg_clf = LogisticRegression(
        solver="saga",
        class_weight="balanced", # Handles class imbalance
        max_iter=4000,           # Increased max_iter for convergence
        random_state=SEED
    )
    log_reg_clf.fit(Z_train_filtered, y_train_adcn)

    # 5. Determine optimal threshold on training data using Youden's J statistic
    # (Sensitivity + Specificity - 1), or equivalently (TPR - FPR)
    train_probas = log_reg_clf.predict_proba(Z_train_filtered)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train_adcn, train_probas)
    optimal_threshold_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_threshold_idx]

    # 6. Calculate metrics
    # Train set
    acc_train = accuracy_score(y_train_adcn, train_probas >= optimal_threshold)
    auc_train = roc_auc_score(y_train_adcn, train_probas)

    # Validation set (using the threshold from the training set)
    val_probas = log_reg_clf.predict_proba(Z_val_filtered)[:, 1]
    acc_val = accuracy_score(y_val_adcn, val_probas >= optimal_threshold)
    auc_val = roc_auc_score(y_val_adcn, val_probas)

    eval_duration_sec = time.time() - eval_start_time
    print(f"  QuickMetrics ({eval_duration_sec:.1f}s) | "
          f"Train AUC: {auc_train:.3f}, ACC: {acc_train:.3f} | "
          f"Val AUC: {auc_val:.3f}, ACC: {acc_val:.3f} (Thresh: {optimal_threshold:.3f})")

    # Optional: Log these specific metrics to W&B if needed during VAE training epochs
    # This is already handled in the train_vae loop, but could be logged here too.
    # if "wandb" in sys.modules and wandb.run is not None:
    #     wandb.log({"Epoch_QuickMetrics_Val_AUC": auc_val, "Epoch_QuickMetrics_Val_ACC": acc_val})

    return acc_train, auc_train, acc_val, auc_val


# ────────────────── Final Classifier Evaluation (using light_pipeline) ─────

def _prepare_adcn_data_for_cv(
    train_X: torch.Tensor, val_X: torch.Tensor,
    train_Y: torch.Tensor, val_Y: torch.Tensor,
    train_L_original: Sequence[str], val_L_original: Sequence[str],
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Prepares data for cross-validation by:
    1. Concatenating train and validation sets.
    2. Filtering for AD (1) and CN (0) classes only.
    3. Returning features (X), labels (y), and subject IDs.

    Args:
        train_X, val_X: Feature tensors for train and validation.
        train_Y, val_Y: Label tensors (0:CN, 1:AD, 2:Other).
        train_L_original, val_L_original: Original string labels/subject IDs.

    Returns:
        Tuple[torch.Tensor, np.ndarray, np.ndarray]:
            - X_combined_adcn: Combined and filtered features.
            - y_combined_adcn: Combined and filtered labels (numpy array).
            - ids_combined_adcn: Combined and filtered subject IDs (numpy array).
    """
    # Combine train and validation sets
    X_combined = torch.cat([train_X, val_X]).cpu() # Move to CPU for consistency
    y_combined_numpy = torch.cat([train_Y, val_Y]).cpu().numpy()
    ids_combined_numpy = np.array(list(train_L_original) + list(val_L_original))

    # Create a mask to keep only AD (1) and CN (0) samples
    adcn_mask = (y_combined_numpy != 2)

    X_combined_adcn = X_combined[adcn_mask]
    y_combined_adcn = y_combined_numpy[adcn_mask]
    ids_combined_adcn = ids_combined_numpy[adcn_mask]

    return X_combined_adcn, y_combined_adcn, ids_combined_adcn


# --- hyperparameters.py ---------------------
import re   # ya estaba importado

def _log_cv_dataframe_means_to_wandb(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Estandariza nombres, elimina duplicados y loguea medias a W&B.
    Devuelve el DF con nombres consistentes.
    """
    # --- 1. Renombrar sin generar duplicados --------------------
    rename_map = {
        "model_base": "model",          #   →  columna “base” será la referencia
        "balanced_accuracy": "bal",
        "f1_score": "f1",
        "pca_applied": "pca",
        "cv_tag": "tag",
    }
    df = df.rename(columns=rename_map)

    # Si por alguna razón todavía viene 'model_config', mantenla aparte
    if "model_config" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"model_config": "model"})

    # --- 2. Garantizar unicidad de nombres ----------------------
    # pandas>=2.1 tiene df.columns.duplicated(); para <2.1 podemos usar:
    duplicated = df.columns.duplicated(keep="first")
    if duplicated.any():
        # Eliminar los duplicados conservando la primera aparición
        df = df.loc[:, ~duplicated]

    # --- 3. Validar columnas clave ------------------------------
    req = {"model", "auc", "bal"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)} en CV-DF (tag={tag})")

    # --- 4. Agrupar y loguear -----------------------------------
    mean_df = df.groupby("model")[["auc", "bal"]].mean()

    for mdl, row in mean_df.iterrows():
        wandb.log({
            f"CV_{tag}_{mdl}_AUC_mean": row.auc,
            f"CV_{tag}_{mdl}_BAL_mean": row.bal,
        })
        wandb.summary[f"CV_{tag}_{mdl}_AUC_mean"] = row.auc
        wandb.summary[f"CV_{tag}_{mdl}_BAL_mean"] = row.bal

    return df




def run_final_classifier_evaluation_cv(
    encoder_model: Encoder, # VAE's encoder part
    train_X: torch.Tensor, val_X: torch.Tensor,
    train_L_original: Sequence[str], val_L_original: Sequence[str],
    train_Y: torch.Tensor, val_Y: torch.Tensor,
    latent_batch_size: int = 256,
) -> float:
    """
    Performs nested Cross-Validation (CV) on the latent features extracted by the encoder.
    Uses functionalities from `light_pipeline.py`.
    Logs detailed CV metrics to W&B. The primary metric returned is the mean AUC
    for Logistic Regression on μ-features, often used for W&B sweeps.

    Args:
        encoder_model (Encoder): The trained encoder module of the VAE.
        train_X, val_X: Feature tensors for training and validation sets.
        train_L_original, val_L_original: Original string labels/IDs for train/val.
        train_Y, val_Y: Encoded label tensors for train/val.
        latent_batch_size (int): Batch size for extracting latent features.

    Returns:
        float: Mean AUC score from Logistic Regression on μ-features from CV.
               Returns 0.0 if `light_pipeline` is unavailable or evaluation fails.
    """
    if not LIGHT_PIPELINE_AVAILABLE:
        print("⚠️ light_pipeline.py is not available. Skipping final classifier CV evaluation.")
        #wandb.log({"CV_mu_LogReg_AUC_mean": 0.0}) # Log a default value
        wandb.summary["CV_mu_LogReg_AUC_mean"] = 0.0
        return 0.0

    print("\n── Starting Final Classifier Cross-Validation Evaluation ──")
    evaluation_start_time = time.time()

    # Ensure encoder is in evaluation mode and on CPU (light_pipeline might expect CPU tensors)
    encoder_cpu = encoder_model.eval().to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Free up GPU memory

    # 1. Prepare combined AD/CN data from train and validation sets
    X_adcn_cv, y_adcn_cv, ids_adcn_cv = _prepare_adcn_data_for_cv(
        train_X, val_X, train_Y, val_Y, train_L_original, val_L_original
    )
    print(f"  Data for CV: X_shape={X_adcn_cv.shape}, y_counts={Counter(y_adcn_cv)}, ids_shape={ids_adcn_cv.shape}")

    # Attempt to get clinical features (if applicable)
    try:
        clinical_features_cv = lp_clinical_onehot_encode_sex(ids_adcn_cv)
        print(f"  Clinical one-hot features shape for CV: {clinical_features_cv.shape}")
    except Exception as e:
        print(f"  Warning: Failed to generate clinical one-hot features for CV. Error: {e}")
        clinical_features_cv = None

    # 2. Iterate over feature types: μ-only and μ+σ (if supported by extract_latent_features)
    primary_auc_metric = 0.0  # This will be CV_mu_LogReg_AUC_mean

    for use_sigma_features, cv_tag in [(False, "mu"), (True, "mu_sigma")]:
        print(f"\n  → Processing CV for tag '{cv_tag}' (with_sigma={use_sigma_features})")
        try:
            # Extract latent features (Z) using light_pipeline's function
            Z_latent_features = lp_extract_latent_features(
                                                                encoder_cpu, X_adcn_cv,
                                                                include_sigma_features=use_sigma_features,  # nuevo nombre
                                                                batch_size=latent_batch_size                # nuevo nombre
                                                            )

            print(f"    Latent matrix Z shape for CV: {Z_latent_features.shape}")

            # Run cross-validation using light_pipeline's run_cross_validation
            # ...
            cv_results_df = lp_run_cross_validation(
                                Z_latent_features, y_adcn_cv,
                                clinical_features=clinical_features_cv,
                                cv_run_tag=cv_tag
                        )

            if cv_results_df.empty:
                raise ValueError("lp_run_cross_validation returned an empty DataFrame.")

            #  ▸ cv_results_df ya viene renombrado
            cv_results_df = _log_cv_dataframe_means_to_wandb(cv_results_df, cv_tag)

            # Ahora esta línea no volverá a fallar
            if (not use_sigma_features) and ("LogReg" in cv_results_df.model.unique()):
                primary_auc_metric = float(
                    cv_results_df.loc[cv_results_df.model == "LogReg", "auc"].mean()
                )
                #wandb.log({f"CV_{cv_tag}_LogReg_AUC_mean": primary_auc_metric})
                wandb.summary["CV_mu_LogReg_AUC_mean"] = primary_auc_metric


        except NotImplementedError as nie:
            print(f"    Skipping tag '{cv_tag}': {nie}") # If extract_latent_features doesn't support sigma
        except Exception as e:
            print(f"    ⚠️ Evaluation for tag '{cv_tag}' failed during CV: {e}")
            # If mu-features LogReg fails, ensure the primary metric is set to 0 for W&B
            if not use_sigma_features:
                #wandb.log({"CV_mu_LogReg_AUC_mean": 0.0})
                wandb.summary["CV_mu_LogReg_AUC_mean"] = 0.0
                primary_auc_metric = 0.0

    # 3. Wrap-up
    evaluation_duration_min = (time.time() - evaluation_start_time) / 60
    print(f"Classifier CV evaluation finished in {evaluation_duration_min:.2f} minutes.")
    return primary_auc_metric


# ────────────────── Test-set Evaluation Utilities ──────────────────

def evaluate_classifiers_on_test_set(
    encoder_model: Encoder,
    train_X: torch.Tensor, val_X: torch.Tensor, test_X: torch.Tensor,
    train_Y: torch.Tensor, val_Y: torch.Tensor, test_Y: torch.Tensor,
    classifier_definitions: Optional[Dict[str, Any]] = None,
    wandb_log_prefix: str = "TEST_",
    variance_threshold: float = 0.0 # Set to 0.0 to disable by default, or use VAR_TH
) -> None:
    """
    Trains classifiers on combined (train+validation) AD/CN data and evaluates them
    on the AD/CN test set. Latent features (μ-only) are extracted from the encoder.

    Args:
        encoder_model (Encoder): The VAE's encoder module.
        train_X, val_X, test_X: Feature tensors for train, validation, and test sets.
        train_Y, val_Y, test_Y: Label tensors (0:CN, 1:AD, 2:Other).
        classifier_definitions (Optional[Dict[str, Any]]): A dictionary defining classifiers
            to train and evaluate. If None, default classifiers (LogReg, SVM, RF, GB) are used.
            Format: {"ClassifierName": (estimator, param_grid, has_predict_proba_method)}
        wandb_log_prefix (str): Prefix for W&B metric logging (e.g., "TEST_").
        variance_threshold (float): Minimum variance for latent features. If > 0,
                                    features with variance below this on the training
                                    data will be removed.
    """
    print("\n── Starting Classifier Evaluation on Test Set ──")
    test_eval_start_time = time.time()

    # 1. Prepare AD/CN data for training (train+val) and testing
    # Training data: Combine train and validation sets, filter for AD/CN
    combined_train_val_Y = torch.cat([train_Y, val_Y])
    adcn_mask_train_val = combined_train_val_Y != 2
    X_train_eval = torch.cat([train_X, val_X])[adcn_mask_train_val]
    y_train_eval = combined_train_val_Y[adcn_mask_train_val].numpy()

    # Test data: Filter for AD/CN
    adcn_mask_test = test_Y != 2
    X_test_eval = test_X[adcn_mask_test]
    y_test_eval = test_Y[adcn_mask_test].numpy()

    if len(np.unique(y_test_eval)) < 2:
        print("⚠️ Test set does not contain both AD and CN classes after filtering. Skipping test set evaluation.")
        return
    if len(np.unique(y_train_eval)) < 2:
        print("⚠️ Combined train/validation set does not contain both AD and CN classes. Skipping test set evaluation.")
        return

    print(f"  Training data for test eval: X_shape={X_train_eval.shape}, y_counts={Counter(y_train_eval)}")
    print(f"  Test data for test eval: X_shape={X_test_eval.shape}, y_counts={Counter(y_test_eval)}")


    # 2. Extract latent features (μ-only)
    # Ensure encoder is in eval mode and on CPU (or DEVICE if lp_extract_latent_features handles it)
    encoder_eval = encoder_model.eval().to(DEVICE) # Use DEVICE for lp_extract_latent_features
                                                # or .to("cpu") if it expects CPU tensors

    # Using the project's extract_latent_features (lp_extract_latent_features)
    if not LIGHT_PIPELINE_AVAILABLE:
        print("⚠️ light_pipeline.py not available. Cannot extract latents for test set evaluation.")
        return

    try:
        Z_train_eval = lp_extract_latent_features(encoder_eval, X_train_eval, include_sigma_features=False, batch_size=256)
        Z_test_eval = lp_extract_latent_features(encoder_eval, X_test_eval, include_sigma_features=False, batch_size=256)
    except Exception as e:
        print(f"Error during latent extraction for test set: {e}. Skipping test evaluation.")
        return


    # 3. Define default classifiers if none provided
    if classifier_definitions is None:
        classifier_definitions = {
            "LogReg": (
                LogisticRegression(solver="saga", class_weight="balanced", max_iter=5000, random_state=SEED),
                {"C": np.logspace(-3, 2, 7), "penalty": ["l1", "l2"]}, True # Has predict_proba
            ),
            "SVM_RBF": (
                SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
                {"C": np.logspace(-1, 2, 8)}, True # Has predict_proba
            ),
            "RF": (
                RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
                {"n_estimators": [100, 300, 600], "max_depth": [None, 10, 20]}, True # Has predict_proba
            ),
            "GB": (
                GradientBoostingClassifier(random_state=SEED),
                {"n_estimators": [100, 300], "learning_rate": [0.01, 0.05, 0.1]}, True # Has predict_proba
            ),
        }

    # 4. Scale features and optionally filter by variance
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train_eval)
    Z_test_scaled = scaler.transform(Z_test_eval)

    if variance_threshold > 0:
        train_variances = np.var(Z_train_scaled, axis=0)
        feature_mask = train_variances > variance_threshold
        
        num_orig_features = Z_train_scaled.shape[1]
        Z_train_final = Z_train_scaled[:, feature_mask]
        Z_test_final = Z_test_scaled[:, feature_mask]
        num_kept = Z_train_final.shape[1]
        print(f"  TestEval Feature Filtering: Kept {num_kept}/{num_orig_features} features (var_thresh={variance_threshold:.1e}).")
        if num_kept == 0:
            print("  TestEval Warning: No features kept after variance filtering for test set. Classifiers may fail or perform poorly.")
            # Potentially return early or handle this case
    else:
        Z_train_final = Z_train_scaled
        Z_test_final = Z_test_scaled
        print(f"  TestEval Feature Filtering: Skipped (variance_threshold={variance_threshold}). Using all {Z_train_final.shape[1]} features.")


    # 5. Train, evaluate, and log each classifier
    # Using HalvingRandomSearchCV for hyperparameter tuning
    inner_cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    for clf_name, (base_estimator, param_dist, has_proba) in classifier_definitions.items():
        print(f"\n  Evaluating {clf_name} on test set...")
        if Z_train_final.shape[1] == 0: # No features to train on
            print(f"    Skipping {clf_name} as there are no features after filtering.")
            # Log neutral/default scores to W&B
            wandb.log({
                f"{wandb_log_prefix}{clf_name}_AUC": 0.5, f"{wandb_log_prefix}{clf_name}_BAL": 0.5,
                f"{wandb_log_prefix}{clf_name}_ACC": 0.5, f"{wandb_log_prefix}{clf_name}_F1": 0.0,
            })
            wandb.summary[f"{wandb_log_prefix}{clf_name}_AUC"] = 0.5
            continue

        try:
            # HalvingRandomSearchCV requires min_resources to be set appropriately
            # _min_resources from light_pipeline or a similar heuristic
            min_res = lp_min_resources(len(y_train_eval)) if LIGHT_PIPELINE_AVAILABLE else 'exhaust'

            search_cv = HalvingRandomSearchCV(
                estimator=base_estimator,
                param_distributions=param_dist,
                cv=inner_cv_strategy,
                scoring="balanced_accuracy", # Or "roc_auc"
                n_candidates='exhaust', # Or a number like 10, 20
                factor=3, # Aggressiveness of halving
                min_resources=min_res,
                random_state=SEED,
                error_score="raise", # Fail loudly
                n_jobs = -1 # Use all available cores for search
            )
            search_cv.fit(Z_train_final, y_train_eval)
            best_clf = search_cv.best_estimator_
            print(f"    Best params for {clf_name}: {search_cv.best_params_}")

            # Predictions and probabilities
            if has_proba:
                test_probas = best_clf.predict_proba(Z_test_final)[:, 1]
                test_preds = (test_probas >= 0.5).astype(int) # Standard 0.5 threshold for probabilities
            else: # For classifiers without predict_proba (e.g., some SVMs)
                test_preds = best_clf.predict(Z_test_final)
                # For AUC, decision_function can be used if available, otherwise AUC might not be directly comparable
                if hasattr(best_clf, 'decision_function'):
                    test_probas = best_clf.decision_function(Z_test_final)
                     # Normalize decision scores to be somewhat like probabilities for roc_auc_score if needed,
                     # though raw scores are fine. For simplicity, if predict_proba is false, AUC might be less standard.
                else: # Cannot calculate AUC without probabilities or decision scores
                    test_probas = test_preds # Fallback, AUC will be based on 0/1 predictions

            # Calculate metrics
            auc_score = roc_auc_score(y_test_eval, test_probas)
            bal_acc_score = balanced_accuracy_score(y_test_eval, test_preds)
            acc_score = accuracy_score(y_test_eval, test_preds)
            f1 = f1_score(y_test_eval, test_preds) # F1 for positive class by default

            # Log metrics to W&B
            wandb.log({
                f"{wandb_log_prefix}{clf_name}_AUC": auc_score,
                f"{wandb_log_prefix}{clf_name}_BAL": bal_acc_score,
                f"{wandb_log_prefix}{clf_name}_ACC": acc_score,
                f"{wandb_log_prefix}{clf_name}_F1": f1,
            })
            wandb.summary[f"{wandb_log_prefix}{clf_name}_AUC"] = auc_score # Update summary too

            print(f"    [TEST RESULTS] {clf_name:10s} | AUC={auc_score:.4f} BAL_ACC={bal_acc_score:.4f} ACC={acc_score:.4f} F1={f1:.4f}")

        except Exception as e:
            print(f"    ⚠️ Classifier {clf_name} failed during test set evaluation: {e}")
            # Log failure or default values
            wandb.log({
                f"{wandb_log_prefix}{clf_name}_AUC": 0.0, f"{wandb_log_prefix}{clf_name}_BAL": 0.0,
                f"{wandb_log_prefix}{clf_name}_ACC": 0.0, f"{wandb_log_prefix}{clf_name}_F1": 0.0,
            })
            wandb.summary[f"{wandb_log_prefix}{clf_name}_AUC"] = 0.0


    test_eval_duration_min = (time.time() - test_eval_start_time) / 60
    print(f"Test set classifier evaluation finished in {test_eval_duration_min:.2f} minutes.")


# ────────────────────── Main Execution Block ────────────────────────────
def main():
    """
    Main function to initialize W&B, load data, train VAE,
    and perform classifier evaluations.
    """
    # Initialize Weights & Biases
    # `reinit=True` allows multiple wandb.init calls in a single script (e.g., for loops)
    # `config=WANDB_DEFAULTS` provides default hyperparameters that can be overridden by a sweep.
# justo al principio de main(), después de wandb.require():
    wandb.init(
        project="beta_vae_classification_sweep",
        config=WANDB_DEFAULTS,
        reinit=True       # sigue funcionando, sólo que está deprecado
    )

    
    # --- Define W&B Metrics ---
    # This helps W&B dashboard to correctly plot and summarize metrics.
    # Define 'epoch' as the primary step metric for VAE training.
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss_total", step_metric="epoch", summary="min")
    wandb.define_metric("val_loss_total", step_metric="epoch", summary="min")
    wandb.define_metric("QuickMetrics_Val_AUC_mu", step_metric="epoch", summary="max") # Quick eval during VAE train

    # Define summary metrics for the entire run (these are often the final important ones)
    wandb.define_metric("best_val_total_loss_VAE", summary="min") # Best VAE validation loss
    
    # Primary metric for sweeps (from final CV evaluation)
    wandb.define_metric("CV_mu_LogReg_AUC_mean", summary="max")

    # Metrics from CV evaluation (means over folds)
    cv_models = ["LogReg", "SVM_RBF", "RF", "GB"] # Add other models if used in light_pipeline
    cv_tags = ["mu", "mu_sigma"]
    for tag in cv_tags:
        for model in cv_models:
            wandb.define_metric(f"CV_{tag}_{model}_AUC_mean", summary="max")
            wandb.define_metric(f"CV_{tag}_{model}_BAL_mean", summary="max")

    # Metrics from final test set evaluation
    test_models = ["LogReg", "SVM_RBF", "RF", "GB"] # From evaluate_classifiers_on_test_set defaults
    for model in test_models:
        wandb.define_metric(f"TEST_{model}_AUC", summary="max")
        wandb.define_metric(f"TEST_{model}_BAL", summary="max")
        wandb.define_metric(f"TEST_{model}_ACC", summary="max")
        wandb.define_metric(f"TEST_{model}_F1", summary="max")

    # Access the effective configuration for this run (includes sweep overrides)
    current_run_config = wandb.config
    print("--- W&B Effective Run Configuration ---")
    for key, val in current_run_config.items():
        print(f"  {key}: {val}")
    # Update global VAR_TH if provided by sweep config, otherwise use the one from settings/fallback
    global VAR_TH 
    VAR_TH = current_run_config.get("VAR_TH", VAR_TH) # Allow sweep to override VAR_TH
    print(f"Effective VAR_TH for this run: {VAR_TH}")


    # --- Data Loading ---
    data_load_start_time = time.time()
    # Using CURRENT_FOLD global variable, can be parameterized if running multiple folds
    trX, vaX, teX, trL, vaL, teL, trY, vaY, teY = load_fold_data(fold_index=CURRENT_FOLD)
    print(f"Data loading completed in: {time.time() - data_load_start_time:.2f} seconds.")

    # --- VAE Training ---
    # train_vae returns the best model and its state_dict
    trained_vae_model, _ = train_vae(current_run_config, trX, vaX, trY, vaY)

    # --- Final Classifier Evaluation using Cross-Validation (on train+val data) ---
    # This uses light_pipeline.py functionalities.
    # The returned `main_sweep_metric` is typically CV_mu_LogReg_AUC_mean.
    main_sweep_metric = run_final_classifier_evaluation_cv(
        encoder_model=trained_vae_model.encoder,
        train_X=trX, val_X=vaX,
        train_L_original=trL, val_L_original=vaL,
        train_Y=trY, val_Y=vaY
    )
    print(f"\nPrimary sweep metric (CV_mu_LogReg_AUC_mean): {main_sweep_metric:.4f}")
    # This metric is already logged and summarized by run_final_classifier_evaluation_cv

    # --- Evaluation on the Hold-Out Test Set ---
    # This trains classifiers on (trX+vaX) and evaluates on teX.
    evaluate_classifiers_on_test_set(
        encoder_model=trained_vae_model.encoder,
        train_X=trX, val_X=vaX, test_X=teX,
        train_Y=trY, val_Y=vaY, test_Y=teY,
        variance_threshold=VAR_TH # Use the effective VAR_TH
    )

    wandb.finish() # Mark the W&B run as finished
    print("\n--- W&B Run Finished Successfully ---")


if __name__ == "__main__":
    # Ensure pandas is available if _log_df_means is used by light_pipeline or other parts
    # import pandas as pd # Already imported at the top
    main()
