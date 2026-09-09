"""Config-driven model factory for Table II architecture/ablation experiments.

Architecture values are stored in ``config.json`` under
``model.model_configs.<experiment>.kwargs``. This module only maps experiment
names to source files, validates the selected config, and instantiates it.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any
import warnings

from model_vanilla import Adaptor_model, ResBlock

__all__ = [
    "Adaptor_model",
    "ResBlock",
    "MODEL_REGISTRY",
    "ARCHITECTURE_MODELS",
    "COMPONENT_VARIANTS",
    "build_model",
    "get_model_identity",
    "get_model_source_path",
    "get_selected_model_config",
    "normalize_model_name",
]


# Only code locations live here. All architecture hyperparameters live in JSON.
MODEL_REGISTRY = {
    "vanilla": ("model_vanilla", "Adaptor_model"),
    "linear": ("model_linear", "Adaptor_model"),
    "mlp": ("model_mlp", "Adaptor_model"),
    "gru": ("model_gru", "Adaptor_model"),
    "tcn": ("model_tcn", "Adaptor_model"),
    "tcn_dynamics": ("model_tcn_dynamics", "Adaptor_model"),
    "tcn_dynamics_v2": ("model_tcn_dynamics_v2", "Adaptor_model"),
    "tcn_dynamics_v3": ("model_tcn_dynamics_v3", "Adaptor_model"),
    "timemixer": ("model_timemixer", "Adaptor_model"),
    "mamba": ("model_mamba", "Adaptor_model"),
    "transformer": ("model_transformer", "Adaptor_model"),
    "lstm": ("model_LSTM", "Adaptor_model"),
    "ours": ("model_tcn_dynamics_v2", "Adaptor_model"),
    "ours_wo_multiscale": ("model_tcn_dynamics_v2", "Adaptor_model"),
    "ours_wo_dilation_kernel_3": ("model_tcn_dynamics_v2", "Adaptor_model"),
    "ours_wo_cagru": ("model_tcn_dynamics_v2", "Adaptor_model"),
    "vanilla_wo_dilation": ("model_vanilla", "Adaptor_model"),
    "vanilla_wo_attention": ("model_vanilla", "Adaptor_model"),
}

ARCHITECTURE_MODELS = (
    "linear",
    "mlp",
    "lstm",
    "timemixer",
    "mamba",
    "transformer",
    "ours",
)
COMPONENT_VARIANTS = (
    "ours_wo_dilation_kernel_3",
    "ours_wo_cagru",
)

_MODEL_NAME_ALIASES = {
    "model pixel level": "vanilla",
    "pixel level": "vanilla",
    "pixel_level": "vanilla",
    "model vanilla": "vanilla",
    "ours": "ours",
    "bi-gru": "gru",
    "bigru": "gru",
    "bi_gru": "gru",
    "ours_wo_multiscale_dilation": "ours_wo_multiscale",
    "ours w/o multiscale dilation": "ours_wo_multiscale",
    "ours_wo_dilation_kernel_3": "ours_wo_dilation_kernel_3",
    "ours w/o dilation kernel 3": "ours_wo_dilation_kernel_3",
    "ours_wo_cagru": "ours_wo_cagru",
    "ours w/o cagru": "ours_wo_cagru",
}


def normalize_model_name(model_name: Any) -> str:
    """Normalize human-readable names to canonical experiment labels."""
    name = str(model_name).strip().lower()
    return _MODEL_NAME_ALIASES.get(name, name)


def _validate_experiment_label(model_config: dict[str, Any]) -> str:
    experiment_label = normalize_model_name(model_config.get("model_name", "vanilla"))
    if experiment_label not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name {model_config.get('model_name')!r}. "
            f"Available models: {', '.join(MODEL_REGISTRY)}"
        )
    return experiment_label


def get_selected_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """Return the selected experiment metadata and explicit constructor kwargs.

    New runs must provide ``model_configs``. A warning-only compatibility path
    remains for historical result configs created before this field existed.
    """
    experiment_label = _validate_experiment_label(model_config)
    all_model_configs = model_config.get("model_configs")
    if all_model_configs is None:
        warnings.warn(
            "Legacy model config has no model.model_configs; constructor defaults "
            "will be used. New experiments should always save explicit configs.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "display_name": experiment_label,
            "experiment_group": "legacy",
            "variant": str(model_config.get("variant", "full")),
            "kwargs": dict(model_config.get("model_kwargs", {}) or {}),
        }

    if not isinstance(all_model_configs, dict):
        raise TypeError("model.model_configs must be a JSON object")
    if experiment_label not in all_model_configs:
        raise ValueError(
            f"model.model_configs has no entry for selected model "
            f"{experiment_label!r}"
        )

    selected = all_model_configs[experiment_label]
    if not isinstance(selected, dict):
        raise TypeError(
            f"model.model_configs.{experiment_label} must be a JSON object"
        )
    if "kwargs" not in selected or not isinstance(selected["kwargs"], dict):
        raise TypeError(
            f"model.model_configs.{experiment_label}.kwargs must be a JSON object"
        )
    return {
        "display_name": str(selected.get("display_name", experiment_label)),
        "experiment_group": str(selected.get("experiment_group", "unspecified")),
        "variant": str(selected.get("variant", "full")),
        "kwargs": dict(selected["kwargs"]),
    }


def get_model_identity(model_config: dict[str, Any]) -> tuple[str, str]:
    """Return the selected experiment label and its recorded variant name."""
    experiment_label = _validate_experiment_label(model_config)
    selected = get_selected_model_config(model_config)
    return experiment_label, selected["variant"]


def build_model(model_config: dict[str, Any]):
    """Instantiate exactly the architecture recorded in ``config.json``."""
    experiment_label = _validate_experiment_label(model_config)
    selected = get_selected_model_config(model_config)
    module_name, class_name = MODEL_REGISTRY[experiment_label]
    module = import_module(module_name)
    model_class = getattr(module, class_name)

    model_kwargs = selected["kwargs"]
    duplicate_shared_keys = {
        "neu_F_dim",
        "latent_F_dim",
        "latent_T_dim",
        "sigma_rescale",
    }.intersection(model_kwargs)
    if duplicate_shared_keys:
        raise ValueError(
            "Put shared settings at model.<key>, not inside model_configs kwargs: "
            f"{sorted(duplicate_shared_keys)}"
        )

    return model_class(
        neu_F_dim=int(model_config["neu_F_dim"]),
        latent_F_dim=int(model_config["latent_F_dim"]),
        latent_T_dim=int(model_config["latent_T_dim"]),
        sigma_rescale=float(model_config["sigma_rescale"]),
        **model_kwargs,
    )


def get_model_source_path(model_config: dict[str, Any]) -> Path:
    """Return the selected architecture source file for result snapshots."""
    experiment_label = _validate_experiment_label(model_config)
    module_name, _ = MODEL_REGISTRY[experiment_label]
    module = import_module(module_name)
    return Path(module.__file__).resolve()


if __name__ == "__main__":
    import json
    import torch

    config_path = Path(__file__).resolve().with_name("config.json")
    with config_path.open(encoding="utf-8") as config_file:
        full_config = json.load(config_file)
    model_config = full_config["model"]
    x = torch.randn(1, 90, int(model_config["neu_F_dim"]))
    for model_name in (*ARCHITECTURE_MODELS, *COMPONENT_VARIANTS):
        selected_config = {**model_config, "model_name": model_name}
        model = build_model(selected_config).eval()
        with torch.no_grad():
            output = model(x)
        expected_shape = (
            1,
            int(model_config["latent_F_dim"]),
            int(model_config["latent_T_dim"]),
        )
        assert output.shape == expected_shape, (model_name, output.shape)
        print(f"{model_name:24s} -> {tuple(output.shape)}")
