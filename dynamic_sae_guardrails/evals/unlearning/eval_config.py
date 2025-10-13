"""Configuration classes for unlearning evaluation.

Defines configurations for different evaluation scenarios:
- UnlearningEvalConfig: WMDP-bio evaluation
- UnlearningEvalConfig_cyber: WMDP-cyber evaluation
- UnlearningEvalConfig_books: MUSE books dataset
- UnlearningEvalConfig_news: MUSE news dataset
"""

from pydantic.dataclasses import dataclass
from pydantic import Field
from evals.base_eval_output import BaseEvalConfig


@dataclass
class UnlearningEvalConfig(BaseEvalConfig):
    """Base configuration for Dynamic SAE Guardrails unlearning evaluation."""
    
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed for reproducibility"
    )
    
    retain_set: str = Field(
        default="wikitext",
        title="Retain Dataset",
        description="Dataset to use for identifying features to retain"
    )
    fgt_set: str = Field(
        default="bio-forget-corpus",
        title="Forget Dataset", 
        description="Dataset containing knowledge to unlearn"
    )

    dataset_names: list[str] = Field(
        default_factory=lambda: [
            "wmdp-bio",
            "high_school_us_history",
            "college_computer_science",
            "high_school_geography",
            "human_aging",
        ],
        title="Dataset Names",
        description="Evaluation datasets: first is unlearning target, rest are retention benchmarks",
    )

    intervention_method: str = Field(
        default="clamp_feature_activation",
        title="Intervention Method",
        description="Intervention method. We only support 'clamp_feature_activation' for now",
    )

    retain_thresholds: list[float] = Field(
        default_factory=lambda: [95],
        title="Retain Thresholds",
        description="Percentile thresholds for filtering features that activate on retain dataset",
    )
    n_features_list: list[int] = Field(
        default_factory=lambda: [20, 50, 100, 200],
        title="N Features List",
        description="Number of top features to intervene on for each configuration",
    )
    multipliers: list[int] = Field(
        default_factory=lambda: [500],
        title="Multipliers",
        description="Negative multiplier values for feature clamping intervention",
    )

    dynamic_thresholds: list[int] = Field(
        default_factory=lambda: [1, 2, 5],
        title="Dynamic Thresholds",
        description="Activation thresholds for dynamic intervention (experimental)"
    )
    
    dataset_fraction: int = Field(
        default=100,
        title="Dataset Fraction",
        description="Percentage of dataset to use for feature sparsity calculation"
    )

    dataset_size: int = Field(
        default=1024,
        title="Dataset Size",
        description="Number of samples for calculating feature sparsity"
    )
    seq_len: int = Field(
        default=1024,
        title="Sequence Length",
        description="Token sequence length for feature sparsity calculation"
    )

    n_batch_loss_added: int = Field(
        default=20,
        title="N Batch Loss Added",
        description="Number of batches for intervention loss calculation (experimental)"
    )
    target_metric: str = Field(
        default="correct",
        title="Target Metric",
        description="Metric type for evaluation: 'correct', 'correct-iff-question', or 'correct-no-tricks'"
    )
    save_metrics: bool = Field(
        default=True,
        title="Save Metrics Flag",
        description="Save metrics for each hyperparameter configuration (required for unlearning score calculation)"
    )

    model_name: str = Field(
        default="",
        title="Model Name",
        description="Model name (set via command line). Recommend instruct-tuned models >= 2B parameters"
    )
    llm_batch_size: int = Field(
        default=None,
        title="LLM Batch Size",
        description="LLM batch size (set automatically or via command line)"
    )
    llm_dtype: str = Field(
        default="",
        title="LLM Data Type",
        description="LLM data type (set automatically or via command line)"
    )


@dataclass
class UnlearningEvalConfigCyber(UnlearningEvalConfig):
    """Configuration for WMDP-cyber unlearning evaluation."""
    
    retain_set: str = Field(default="wikitext")
    fgt_set: str = Field(default="cyber-forget-corpus")

    dataset_names: list[str] = Field(
        default_factory=lambda: [
            "wmdp-cyber",
            "high_school_us_history",
            "college_biology",
            "high_school_geography",
            "human_aging",
        ],
        title="Dataset Names",
        description="List of dataset names. We want to unlearn wmdp-bio while retaining knowledge in other datasets",
    )

    intervention_method: str = Field(
        default="clamp_feature_activation",
        title="Intervention Method",
        description="Intervention method. We only support 'clamp_feature_activation' for now",
    )

    retain_thresholds: list[float] = Field(
        default_factory=lambda: [90],
        title="Retain Thresholds",
        description="We ignore features that activate more than this threshold on the retain dataset",
    )
    n_features_list: list[int] = Field(
        default_factory=lambda: [30],
        title="N Features List",
        description="Each N is the number of features we select and clamp to a negative value",
    )
    multipliers: list[int] = Field(
        default_factory=lambda: [500],
        title="Multipliers",
        description="A list of negative values. We iterate over this list, clamping the selected features to each value",
    )

    # Dynamic thresholds (currently not used)
    dynamic_thresholds: list[int] = Field(
        default_factory=lambda: [1,2,5],
        title="Dynamic Thresholds",
        description="A list of dynamic thresholds to apply",
    )
    
    dataset_fraction: int = Field(
        default=100,
        title="Dataset Size",
        description="Dataset size we use when calculating feature sparsity",
    )


@dataclass
class UnlearningEvalConfigBooks(UnlearningEvalConfig):
    """
    NOTE: This config only identifies SAE features for MUSE-Books dataset.
    Actual MUSE unlearning evaluation is performed using: https://github.com/locuslab/open-unlearning
    """
    
    retain_set: str = Field(default="books")
    fgt_set: str = Field(default="books")

    dataset_names: list[str] = Field(
        default_factory=lambda: [
            "wmdp-cyber",
            "high_school_us_history",
            "college_biology",
            "high_school_geography",
            "human_aging",
        ],
        title="Dataset Names",
        description="List of dataset names. We want to unlearn wmdp-bio while retaining knowledge in other datasets",
    )

    intervention_method: str = Field(
        default="clamp_feature_activation",
        title="Intervention Method",
        description="Intervention method. We only support 'clamp_feature_activation' for now",
    )

    retain_thresholds: list[float] = Field(
        default_factory=lambda: [95, 90],
        title="Retain Thresholds",
        description="We ignore features that activate more than this threshold on the retain dataset",
    )
    n_features_list: list[int] = Field(
        default_factory=lambda: [10, 20, 30],
        title="N Features List",
        description="Each N is the number of features we select and clamp to a negative value",
    )
    multipliers: list[int] = Field(
        default_factory=lambda: [500],#[10,50,100,200, 250, 300, 350, 500],
        title="Multipliers",
        description="A list of negative values. We iterate over this list, clamping the selected features to each value",
    )
@dataclass
class UnlearningEvalConfigNews(UnlearningEvalConfig):
    """
    NOTE: This config only identifies SAE features for MUSE-News dataset.
    Actual MUSE unlearning evaluation is performed using: https://github.com/locuslab/open-unlearning
    """
    
    retain_set: str = Field(default="news")
    fgt_set: str = Field(default="news")

    dataset_names: list[str] = Field(
        default_factory=lambda: [
            "wmdp-cyber",
            "high_school_us_history",
            "college_biology",
            "high_school_geography",
            "human_aging",
        ],
        title="Dataset Names",
        description="List of dataset names. We want to unlearn wmdp-bio while retaining knowledge in other datasets",
    )

    intervention_method: str = Field(
        default="clamp_feature_activation",
        title="Intervention Method",
        description="Intervention method. We only support 'clamp_feature_activation' for now",
    )

    retain_thresholds: list[float] = Field(
        default_factory=lambda: [90],#,80,70,60],#, 90, 85, 80, 70, 60, 50], #[0.01, 0.001, 0.1, 1], #[0.001, 0.01],
        title="Retain Thresholds",
        description="We ignore features that activate more than this threshold on the retain dataset",
    )
    n_features_list: list[int] = Field(
        default_factory=lambda: [10,20,30], #[20, 50, 100, 200, 500, 1000, 10000], # Not used
        title="N Features List",
        description="Each N is the number of features we select and clamp to a negative value",
    )
    multipliers: list[int] = Field(
        default_factory=lambda: [500],#[10,50,100,200, 250, 300, 350, 500],
        title="Multipliers",
        description="A list of negative values. We iterate over this list, clamping the selected features to each value",
    )