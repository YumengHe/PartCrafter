import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
import trimesh
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from ..utils.inference_utils import hierarchical_extract_geometry

from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import PartCrafterDiTModel
from .pipeline_partcrafter_output import PartCrafterPipelineOutput
from .pipeline_utils import TransformerDiffusionMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class PartCrafterPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Pipeline for image to 3D part-level object generation.       
    """

    def __init__(
        self,   
        vae: TripoSGVAEModel,
        transformer: PartCrafterDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        global_image: PipelineImageInput = None,
        local_images: PipelineImageInput = None,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Handle backward compatibility and determine batch size
        if global_image is None and local_images is None:
            # Backward compatibility: use image parameter for both global and local
            global_image = image
            local_images = image

        if global_image is not None:
            if isinstance(global_image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(global_image, list):
                batch_size = len(global_image)
            elif isinstance(global_image, torch.Tensor):
                batch_size = global_image.shape[0]
            else:
                raise ValueError("Invalid input type for global_image")
        elif local_images is not None:
            if isinstance(local_images, PIL.Image.Image):
                batch_size = 1
            elif isinstance(local_images, list):
                batch_size = len(local_images)
            elif isinstance(local_images, torch.Tensor):
                batch_size = local_images.shape[0]
            else:
                raise ValueError("Invalid input type for local_images")
        else:
            raise ValueError("Either global_image or local_images must be provided")

        device = self._execution_device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode conditions separately
        global_image_embeds = None
        global_negative_image_embeds = None
        local_image_embeds = None
        local_negative_image_embeds = None

        if global_image is not None:
            global_image_embeds, global_negative_image_embeds = self.encode_image(
                global_image, device, num_images_per_prompt
            )

        if local_images is not None:
            local_image_embeds, local_negative_image_embeds = self.encode_image(
                local_images, device, num_images_per_prompt
            )

        # For backward compatibility, set image_embeds to local_image_embeds at the beginning
        image_embeds = local_image_embeds
        negative_image_embeds = local_negative_image_embeds

        # Handle dimension mismatch: if global_image is single but local_images are multiple,
        # expand global image embeds to match local images dimensions
        if global_image_embeds is not None and local_image_embeds is not None:
            if global_image_embeds.shape[0] != local_image_embeds.shape[0]:
                if global_image_embeds.shape[0] == 1 and local_image_embeds.shape[0] > 1:
                    # Expand single global image to match multiple local images
                    num_local_images = local_image_embeds.shape[0]
                    global_image_embeds = global_image_embeds.repeat(num_local_images, 1, 1)
                    global_negative_image_embeds = global_negative_image_embeds.repeat(num_local_images, 1, 1)
                elif global_image_embeds.shape[0] > 1 and local_image_embeds.shape[0] == 1:
                    # This case shouldn't happen in normal usage, but handle it for completeness
                    local_image_embeds = local_image_embeds.repeat(global_image_embeds.shape[0], 1, 1)
                    local_negative_image_embeds = local_negative_image_embeds.repeat(global_image_embeds.shape[0], 1, 1)
                    # Update backward compatibility variables
                    image_embeds = local_image_embeds
                    negative_image_embeds = local_negative_image_embeds
                else:
                    raise ValueError(f"Dimension mismatch between global_image ({global_image_embeds.shape[0]}) and local_images ({local_image_embeds.shape[0]})")


        # Verify dimension consistency for debugging
        if global_image_embeds is not None and local_image_embeds is not None:
            print(f"DEBUG Pipeline: Global image embeds shape: {global_image_embeds.shape}")
            print(f"DEBUG Pipeline: Local image embeds shape: {local_image_embeds.shape}")
            assert global_image_embeds.shape[0] == local_image_embeds.shape[0], \
                f"Global and local image embeds dimension mismatch: global={global_image_embeds.shape}, local={local_image_embeds.shape}"

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)
            if global_image_embeds is not None:
                global_image_embeds = torch.cat([global_negative_image_embeds, global_image_embeds], dim=0)
            if local_image_embeds is not None:
                local_image_embeds = torch.cat([local_negative_image_embeds, local_image_embeds], dim=0)

            # Debug CFG shapes
            if global_image_embeds is not None and local_image_embeds is not None:
                print(f"DEBUG Pipeline CFG: Global image embeds shape after CFG: {global_image_embeds.shape}")
                print(f"DEBUG Pipeline CFG: Local image embeds shape after CFG: {local_image_embeds.shape}")
                assert global_image_embeds.shape[0] == local_image_embeds.shape[0], \
                    f"CFG: Global and local image embeds dimension mismatch: global={global_image_embeds.shape}, local={local_image_embeds.shape}"

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        self.set_progress_bar_config(
            desc="Denoising", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    global_encoder_hidden_states=global_image_embeds,
                    local_encoder_hidden_states=local_image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0].to(dtype)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    image_embeds_1 = callback_outputs.pop(
                        "image_embeds_1", image_embeds_1
                    )
                    negative_image_embeds_1 = callback_outputs.pop(
                        "negative_image_embeds_1", negative_image_embeds_1
                    )
                    image_embeds_2 = callback_outputs.pop(
                        "image_embeds_2", image_embeds_2
                    )
                    negative_image_embeds_2 = callback_outputs.pop(
                        "negative_image_embeds_2", negative_image_embeds_2
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()


        # 7. decoder mesh
        self.vae.set_flash_decoder()
        output, meshes = [], []
        self.set_progress_bar_config(
            desc="Decoding", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=batch_size) as progress_bar:
            for i in range(batch_size):
                geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                        # verbose=True
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()
       
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return PartCrafterPipelineOutput(samples=output, meshes=meshes)

