"""Feature intervention utilities for unlearning evaluation.

This module implements the dynamic intervention method for clamping SAE features
during evaluation, with activation threshold-based filtering.
"""

import torch
from jaxtyping import Float
from sae_lens import SAE
from torch import Tensor
from transformer_lens.hook_points import HookPoint


def anthropic_clamp_resid_SAE_features(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    sae: SAE,
    features_to_ablate: list[int],
    multiplier: float,
    activation_threshold: float,
    random: bool = False,
) -> Float[Tensor, "batch seq d_model"]:
    """Apply dynamic feature clamping with activation threshold filtering.
    
    Enhanced version of Anthropic's feature clamping that adds dynamic filtering
    based on activation patterns. Only intervenes on batches where feature
    activation rates exceed the specified threshold.
    
    Args:
        resid: Residual stream activations
        hook: Hook point (unused but required for hook interface)
        sae: Sparse autoencoder
        features_to_ablate: List of feature indices to clamp
        multiplier: Clamping value (negative to suppress features)
        activation_threshold: Threshold for batch-level intervention
        random: Whether to use random ablation (deprecated)
        
    Returns:
        Modified residual stream with clamped features
    """
    if len(features_to_ablate) > 0:
        # Encode residual stream and zero out BOS token
        feature_activations = sae.encode(resid)
        feature_activations[:, 0, :] = 0.0
        
        # Get initial reconstruction and error term
        reconstruction = sae.decode(feature_activations)
        error = resid - reconstruction

        # Create mask for features that exceed activation threshold
        target_features = feature_activations[:, :, features_to_ablate]
        activation_mask = target_features > 0
        activation_mask = (activation_mask.sum(dim=2) > 0)
        
        # Calculate which batches exceed the activation threshold
        batch_activation_rates = activation_mask.sum(dim=1) / activation_mask.shape[1]
        active_batches = batch_activation_rates > activation_threshold

        # Create final mask combining feature activation and batch threshold
        final_mask = activation_mask.unsqueeze(2) & active_batches.unsqueeze(1).unsqueeze(2)
        
        # Apply clamping to selected features
        feature_activations[:, :, features_to_ablate] = torch.where(
            final_mask,
            torch.full_like(target_features, -multiplier),
            feature_activations[:, :, features_to_ablate]
        )

        # Reconstruct and add back error term
        modified_reconstruction = sae.decode(feature_activations)
        resid = modified_reconstruction + error

    return resid
