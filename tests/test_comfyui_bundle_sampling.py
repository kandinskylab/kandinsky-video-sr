"""ComfyUI SR sampling settings come from the model's Diffusers bundle, not from constants in the node pack.

What & why: ``Load Diffusion Model`` gives ComfyUI only the transformer weights, so
the pi-Flow shift used to be a release constant (3.5). A bundle with another
shift (VSR 2.0 ships 5.0) then sampled on the wrong schedule. The settings must
be read from ``scheduler/scheduler_config.json`` and ``transformer/config.json``
next to the weights. How: unit tests on ``kandinsky6_vsr.bundle_sampling`` with
the tiny bundle factory from ``conftest``; no ComfyUI, weights or CUDA. Corner
cases: a non-default shift is honoured, a flow-matching bundle yields the Euler
settings only, a missing scheduler config or ``config.json`` fails loudly, a
DiT whose output head disagrees with the bundle grid is rejected, and a model
loaded without bundle settings cannot be sampled.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kandinsky6_vsr.bundle_sampling import attach_bundle_sampling, bundle_sampling, read_bundle_sampling

from .conftest import FLOW_MATCHING_SHIFT, PIFLOW_GRID_POINTS, PIFLOW_SHIFT, WEIGHTS_NAME, BundleFactory

VSR2_SHIFT = 5.0  # the shift VSR 2.0 ships with, unlike the 3.5 of VSR 1.0


def model_patcher(n_grid: int) -> SimpleNamespace:
    """A stand-in for a ComfyUI ``ModelPatcher`` wrapping the SR DiT."""
    return SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(n_grid=n_grid)))


def set_scheduler_field(bundle: Path, name: str, value: float) -> None:
    config_file = bundle / "scheduler" / "scheduler_config.json"
    config = json.loads(config_file.read_text())
    config_file.write_text(json.dumps({**config, name: value}))


def test_piflow_settings_follow_the_scheduler_config(make_bundle: BundleFactory) -> None:
    settings = read_bundle_sampling(make_bundle("piflow") / "transformer")
    assert settings == {
        "lq_noise_scale": 0.7,
        "lq_noise_type": "ddpm",
        "scheduler_scale": PIFLOW_SHIFT,
        "nfe": 2,
        "n_grid": PIFLOW_GRID_POINTS,
        "shift": PIFLOW_SHIFT,
        "eps": 1e-06,
        "final_step_size_scale": 0.5,
        "num_policy_substeps": 128,
    }


def test_a_non_default_shift_is_not_overridden(make_bundle: BundleFactory) -> None:
    bundle = make_bundle("piflow")
    set_scheduler_field(bundle, "shift", VSR2_SHIFT)
    assert read_bundle_sampling(bundle / "transformer")["shift"] == VSR2_SHIFT


def test_flow_matching_bundle_yields_euler_settings(make_bundle: BundleFactory) -> None:
    settings = read_bundle_sampling(make_bundle("flow_matching") / "transformer")
    assert settings == {"lq_noise_scale": 0.7, "lq_noise_type": "ddpm", "scheduler_scale": FLOW_MATCHING_SHIFT}


def test_missing_scheduler_config_fails(make_bundle: BundleFactory) -> None:
    bundle = make_bundle("piflow")
    (bundle / "scheduler" / "scheduler_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="scheduler_config.json"):
        read_bundle_sampling(bundle / "transformer")


def test_weights_without_their_config_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.json"):
        read_bundle_sampling(tmp_path)


def test_attach_reads_the_bundle_next_to_the_weights(make_bundle: BundleFactory) -> None:
    patcher = model_patcher(PIFLOW_GRID_POINTS)
    attach_bundle_sampling(patcher, make_bundle("piflow") / "transformer" / WEIGHTS_NAME)
    assert bundle_sampling(patcher)["shift"] == PIFLOW_SHIFT


def test_attach_rejects_a_grid_mismatch(make_bundle: BundleFactory) -> None:
    with pytest.raises(ValueError, match="n_grid"):
        attach_bundle_sampling(model_patcher(1), make_bundle("piflow") / "transformer" / WEIGHTS_NAME)


def test_sampling_without_bundle_settings_fails() -> None:
    with pytest.raises(ValueError, match="scheduler_config.json"):
        bundle_sampling(model_patcher(PIFLOW_GRID_POINTS))
