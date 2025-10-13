"""Evaluation utilities for unlearning experiments."""

import os

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from evals.unlearning.utils.feature_activation import (
    get_top_features,
    get_top_features_ratio,
    get_top_features_percentile,
    get_top_features_threshold,
    load_sparsity_data,
    save_feature_sparsity,
    get_shuffled_forget_retain_tokens,
)
from evals.unlearning.utils.metrics import calculate_metrics_list
from evals.unlearning.eval_config import UnlearningEvalConfig

def run_metrics_calculation(
    model: HookedTransformer,
    sae: SAE,
    activation_store,
    forget_sparsity: np.ndarray,
    retain_sparsity: np.ndarray,
    artifacts_folder: str,
    sae_name: str,
    config: UnlearningEvalConfig,
    force_rerun: bool,
):
    """Calculate unlearning metrics for various hyperparameter combinations.
    
    Args:
        model: The transformer model
        sae: Sparse autoencoder
        activation_store: Storage for activations (unused)
        forget_sparsity: Feature sparsity on forget dataset
        retain_sparsity: Feature sparsity on retain dataset
        artifacts_folder: Directory for saving artifacts
        sae_name: Name of the SAE being evaluated
        config: Evaluation configuration
        force_rerun: Whether to force recalculation
        
    Returns:
        list: Evaluation metrics for each hyperparameter combination
    """
    dataset_names = config.dataset_names
    
    folder_name = os.path.join(artifacts_folder, sae_name, "results", "sparsities")
    for retain_threshold in config.retain_thresholds:
        # Get top features based on percentile thresholds
        top_features_custom, threshold_inf = get_top_features_percentile(
            forget_sparsity, 
            retain_sparsity, 
            ratio_percentile=retain_threshold,
            folder_name=folder_name,
            n_features_lst=config.n_features_list
        )

        main_ablate_params = {
            "intervention_method": config.intervention_method,
            
        }

        n_features_lst = config.n_features_list
        multipliers = config.multipliers

        sweep = {
            "features_to_ablate": [np.array(top_features_custom[:n]) for n in n_features_lst],
            "multiplier": multipliers,
            'activation_threshold': threshold_inf,
        }


        save_metrics_dir = os.path.join(artifacts_folder, sae_name, "results/metrics")

        metrics_lst = calculate_metrics_list(
            model,
            config.llm_batch_size,
            sae,
            main_ablate_params,
            sweep,
            artifacts_folder,
            force_rerun,
            dataset_names,
            n_batch_loss_added=config.n_batch_loss_added,
            activation_store=activation_store,
            target_metric=config.target_metric,
            save_metrics=config.save_metrics,
            save_metrics_dir=save_metrics_dir,
            retain_threshold=retain_threshold,
            seed=config.random_seed,
        )

    return metrics_lst

def compute_params_SAE(
    model: HookedTransformer,
    sae: SAE,
    activation_store,
    forget_sparsity: np.ndarray,
    retain_sparsity: np.ndarray,
    artifacts_folder: str,
    sae_name: str,
    config: UnlearningEvalConfig,
    force_rerun: bool,
):
    """Compute SAE parameters for MUSE datasets without running full evaluation.
    
    This function identifies important features based on activation patterns
    but does not perform intervention or evaluation metrics calculation.
    Used specifically for MUSE (books/news) datasets.
    
    Args:
        model: The transformer model
        sae: Sparse autoencoder
        activation_store: Storage for activations (unused)
        forget_sparsity: Feature sparsity on forget dataset
        retain_sparsity: Feature sparsity on retain dataset
        artifacts_folder: Directory for saving artifacts
        sae_name: Name of the SAE being evaluated
        config: Evaluation configuration
        force_rerun: Whether to force recalculation
    """
    dataset_names = config.dataset_names
    folder_name = os.path.join(artifacts_folder, sae_name, "results", "sparsities")
    for retain_threshold in config.retain_thresholds:

        top_features_custom, threshold_inf = get_top_features_percentile(
            forget_sparsity, 
            retain_sparsity, 
            ratio_percentile=retain_threshold,
            folder_name=folder_name,
            n_features_lst=config.n_features_list
        )
        
        main_ablate_params = {
            "intervention_method": config.intervention_method,
            
        }

        n_features_lst = config.n_features_list
        multipliers = config.multipliers

        sweep = {
            'threshold': retain_threshold,
            "features_to_ablate": [np.array(top_features_custom[:n]) for n in n_features_lst],
            "multiplier": multipliers,
            'activation_threshold': threshold_inf,
        }
        print(f"Hyperparameter sweep configuration: {sweep}")



def run_eval_single_sae(
    model: HookedTransformer,
    sae: SAE,
    config: UnlearningEvalConfig,
    artifacts_folder: str,
    sae_release_and_id: str,
    force_rerun: bool,
):
    """Run unlearning evaluation on a single SAE.
    
    This function orchestrates the full evaluation pipeline:
    1. Calculate feature sparsity on forget/retain datasets
    2. Run metrics calculation (for WMDP) or parameter computation (for MUSE)
    
    Args:
        model: The transformer model
        sae: Sparse autoencoder to evaluate
        config: Evaluation configuration
        artifacts_folder: Directory for saving intermediate results
        sae_release_and_id: Unique identifier for this SAE (used for caching)
        force_rerun: Whether to force recalculation even if cached results exist
    """

    os.makedirs(artifacts_folder, exist_ok=True)

    torch.set_grad_enabled(False)

    # Step 1: Calculate feature sparsity on forget and retain datasets
    save_feature_sparsity(
        model,
        sae,
        artifacts_folder,
        sae_release_and_id,
        config.dataset_size,
        config.seq_len,
        config.llm_batch_size,
        config.dataset_fraction,
        fgt_set=config.fgt_set,
        retain_set=config.retain_set
    )
    forget_sparsity, retain_sparsity = load_sparsity_data(artifacts_folder, sae_release_and_id)
    
    # Activation store not needed for current implementation
    activation_store = None
    
    # Step 2: Run evaluation based on dataset type
    # IMPORTANT: MUSE (books/news) evaluation differs from WMDP evaluation
    # - MUSE: Only identifies important features based on activation patterns, 
    #   does NOT evaluate actual unlearning performance
    # - WMDP: Full evaluation pipeline including interventions and performance metrics
    if config.fgt_set == 'books' or config.fgt_set == 'news':
        # For MUSE datasets, only compute SAE parameters without evaluation
        compute_params_SAE(  
            model,
            sae,
            activation_store,
            forget_sparsity,
            retain_sparsity,
            artifacts_folder,
            sae_release_and_id,
            config,
            force_rerun,
        )
    else:
        results = run_metrics_calculation(
            model,
            sae,
            activation_store,
            forget_sparsity,
            retain_sparsity,
            artifacts_folder,
            sae_release_and_id,
            config,
            force_rerun,
        )

        return results
