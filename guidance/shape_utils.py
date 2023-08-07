from typing import Any, Dict, Optional, Sequence, Tuple, Union
from shap_e.models.transmitter.base import Transmitter
from shap_e.models.query import Query
from shap_e.models.nerstf.renderer import NeRSTFRenderer
from shap_e.util.collections import AttrDict
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.image_util import load_image
from shap_e.models.nn.meta import subdict
import torch
import gc


camera_to_shapes = [
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32),
        torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32), # to bird view
        torch.tensor([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=torch.float32), # to rotaed bird view
        torch.tensor([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=torch.float32), # to rotaed bird view
        torch.tensor([[-1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32), # to bird view
        ]


def get_density(
    render,
    query: Query,
    params: Dict[str, torch.Tensor],
    options: AttrDict[str, Any],
) -> torch.Tensor:
    assert render.nerstf is not None
    return render.nerstf(query, params=subdict(params, "nerstf"), options=options).density


@torch.no_grad()
def get_shape_from_image(image_path, pos, 
                             rpst_type='sdf', # or 'density'
                             get_color=True, 
                             shape_guidance=3, device='cuda'):
    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    latent = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=shape_guidance,
        model_kwargs=dict(images=[load_image(image_path)]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )[0]
    
    params = (xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        )

    rpsts, colors = [], []
    for camera_to_shape in camera_to_shapes:
        query = Query(
            position=pos @ camera_to_shape.to(pos.device),
            direction=None,
        )
        
        if rpst_type == 'sdf': 
            rpst = xm.renderer.get_signed_distance(query, params, AttrDict()) 
        else:
            rpst = get_density(xm.renderer, query, params, AttrDict()) 
        rpsts.append(rpst.squeeze())

        if get_color:
            color = xm.renderer.get_texture(query, params, AttrDict()) 
        else:
            color = None
        colors.append(color)

    return rpsts, colors