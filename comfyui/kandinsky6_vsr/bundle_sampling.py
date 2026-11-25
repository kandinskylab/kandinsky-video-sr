"""Sampling settings of a Kandinsky 6 SR DiT, read from its Diffusers bundle.

``Load Diffusion Model`` hands ComfyUI only the transformer weights. The sampler
settings — the pi-Flow grid, shift and segment schedule, the flow-matching
shift, the noise of the degraded start — belong to the checkpoint and live next
to those weights in ``transformer/config.json`` and the sibling
``scheduler/scheduler_config.json``. They are read at load time and kept on the
diffusion model, so every SR run samples the way that checkpoint was trained.
"""

from pathlib import Path

from .runtime.pipeline.diffusers_bundle import COMPONENT_CONFIG, training_config_from_transformer

_SETTINGS_ATTRIBUTE = "k6_sr_sampling"
_START_NOISE_PARAMS = ("lq_noise_scale", "lq_noise_type", "scheduler_scale")


def read_bundle_sampling(transformer_dir):
    """Read the sampling settings of the SR DiT in a bundle's ``transformer`` component.

    Args:
        transformer_dir: Directory holding ``config.json`` and the DiT weights.

    Returns:
        ``lq_noise_scale``, ``lq_noise_type`` and ``scheduler_scale`` (the Euler
        shift); a pi-Flow checkpoint adds ``nfe``, ``n_grid``, ``shift``, ``eps``,
        ``final_step_size_scale`` and ``num_policy_substeps``.

    Raises:
        FileNotFoundError: If ``config.json`` or the sibling scheduler config is missing.
        ValueError: If the scheduler is neither pi-Flow nor flow-matching Euler.
    """
    transformer_dir = Path(transformer_dir)
    if not (transformer_dir / COMPONENT_CONFIG).is_file():
        raise FileNotFoundError(
            f"{transformer_dir / COMPONENT_CONFIG} is missing: keep the SR DiT inside its Diffusers bundle "
            "(transformer/config.json and scheduler/scheduler_config.json beside the weights)."
        )
    trainer = training_config_from_transformer(transformer_dir).trainer
    settings = {name: trainer.params[name] for name in _START_NOISE_PARAMS}
    if "piflow" in trainer:
        piflow = trainer.piflow
        settings |= {
            "nfe": int(piflow.nfe),
            "n_grid": int(piflow.dx_num_grid_points),
            "shift": float(piflow.shift),
            "eps": float(piflow.eps),
            "final_step_size_scale": float(piflow.final_step_size_scale),
            "num_policy_substeps": int(piflow.num_policy_substeps),
        }
    return settings


def attach_bundle_sampling(model_patcher, weights_path):
    """Read the settings of the bundle holding ``weights_path`` and keep them on the loaded DiT.

    Raises:
        ValueError: If the DiT's output head and the bundle's pi-Flow grid disagree.
    """
    settings = read_bundle_sampling(Path(weights_path).parent)
    diffusion_model = model_patcher.model.diffusion_model
    if int(settings.get("n_grid", 1)) != int(diffusion_model.n_grid):
        raise ValueError(
            f"{weights_path}: the DiT has n_grid={diffusion_model.n_grid}, its scheduler config "
            f"n_grid={settings.get('n_grid', 1)}; the weights and the bundle configs do not belong together."
        )
    setattr(diffusion_model, _SETTINGS_ATTRIBUTE, settings)


def bundle_sampling(model_patcher):
    """The sampling settings attached to an SR DiT at load time.

    Raises:
        ValueError: If the model was not loaded from a Diffusers bundle.
    """
    settings = getattr(model_patcher.model.diffusion_model, _SETTINGS_ATTRIBUTE, None)
    if settings is None:
        raise ValueError(
            "The Kandinsky6 SR DiT carries no sampling settings: load it with Load Diffusion Model from "
            "<bundle>/transformer/diffusion_pytorch_model.safetensors, with transformer/config.json and "
            "scheduler/scheduler_config.json in place."
        )
    return settings
