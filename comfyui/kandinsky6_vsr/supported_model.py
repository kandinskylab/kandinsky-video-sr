"""ComfyUI supported-model registration of the Kandinsky 6 SR DiT."""

import comfy.supported_models_base
import torch

from . import model_base
from .checkpoint_keys import native_dit_state

SR_IMAGE_MODEL = "kandinsky6_sr"


class Kandinsky6SR(comfy.supported_models_base.BASE):
    unet_config = {"image_model": SR_IMAGE_MODEL}
    unet_extra_config = {}
    latent_format = model_base.Kandinsky6SRLatent
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    memory_usage_factor = 1.0
    # No sampling_settings: the SR samplers build their own schedule from the
    # bundle's scheduler config (see bundle_sampling), not from model_sampling.

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.Kandinsky6SRModel(self, device=device)

    def process_unet_state_dict(self, state_dict):
        return native_dit_state(state_dict)
