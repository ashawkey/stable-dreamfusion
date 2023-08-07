from typing import List, Optional, Sequence, Tuple, Union, Mapping
import os

from dataclasses import dataclass
from torch.cuda.amp import custom_bwd, custom_fwd
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.utils import make_grid, save_image
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPProcessor
import logging
logger = logging.getLogger(__name__)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def seed_everything(seed=None):
    if seed:
        seed = int(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def save_tensor2image(x: torch.Tensor, path, channel_last=True, quality=75, **kwargs):
    # assume the input x is channel last
    if x.ndim == 4 and channel_last:
        x = x.permute(0, 3, 1, 2) 
    TVF.to_pil_image(make_grid(x, value_range=(0, 1), **kwargs)).save(path, quality=quality)


def to_pil(x: torch.Tensor, **kwargs) -> Image.Image:
    return TVF.to_pil_image(make_grid(x, value_range=(0, 1), **kwargs))


def to_np_img(x: torch.Tensor) -> np.ndarray:
    return (x.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def token_replace(prompt, negative, learned_embeds_path):
    # Set up automatic token replacement for prompt
    if '<token>' in prompt or '<token>' in negative:
        if learned_embeds_path is None:
            raise ValueError(
                '--learned_embeds_path must be specified when using <token>')
        import torch
        tmp = list(torch.load(learned_embeds_path, map_location='cpu').keys())
        if len(tmp) != 1:
            raise ValueError(
                'Something is wrong with the dict passed in for --learned_embeds_path')
        token = tmp[0]
        prompt = prompt.replace('<token>', token)
        negative = negative.replace('<token>', token)
        logger.info(f'Prompt after replacing <token>: {prompt}')
        logger.info(f'Negative prompt after replacing <token>: {negative}')
    return prompt, negative


@dataclass
class UNet2DConditionOutput:
    # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor
    sample: torch.HalfTensor


def enable_vram(pipe):
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.enable_attention_slicing(1)
    # pipe.enable_model_cpu_offload()


def get_model_path(sd_version='2.1', clip_version='large', hf_key=None):
    if hf_key is not None:
        logger.info(f'[INFO] using hugging face custom model key: {hf_key}')
        sd_path = hf_key
    elif sd_version == '2.1':
        sd_path = "stabilityai/stable-diffusion-2-1-base"
    elif sd_version == '2.0':
        sd_path = "stabilityai/stable-diffusion-2-base"
    elif sd_version == '1.5':
        sd_path = "runwayml/stable-diffusion-v1-5"
    else:
        raise ValueError(
            f'Stable-diffusion version {sd_version} not supported.')
    if clip_version == 'base':
        clip_path = "openai/clip-vit-base-patch32"
    else:
        clip_path = "openai/clip-vit-large-patch14"
    return sd_path, clip_path


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O,
                 sd_version='2.1', hf_key=None,
                 t_range=[0.02, 0.98],
                 use_clip=False,
                 clip_version='base',
                 clip_iterative=True,
                 clip_t=0.4,
                 **kwargs
                 ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.vram_O = vram_O
        self.fp16 = fp16

        logger.info(f'[INFO] loading stable diffusion...')

        sd_path, clip_path = get_model_path(sd_version, clip_version, hf_key)
        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_path, torch_dtype=self.precision_t, local_files_only=False)

        if isfile('./unet_traced.pt'):
            # use jitted unet
            unet_traced = torch.jit.load('./unet_traced.pt')

            class TracedUNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_channels = pipe.unet.in_channels
                    self.device = pipe.unet.device

                def forward(self, latent_model_input, t, encoder_hidden_states):
                    sample = unet_traced(
                        latent_model_input, t, encoder_hidden_states)[0]
                    return UNet2DConditionOutput(sample=sample)
            pipe.unet = TracedUNet()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        if kwargs.get('learned_embeds_path', None) is not None:
            learned_embeds_path = kwargs['learned_embeds_path']
            if os.path.exists(learned_embeds_path):
                logger.info(
                    f'[INFO] loading learned embeddings from {kwargs["learned_embeds_path"]}')
                self.add_tokens_to_model_from_path(learned_embeds_path, kwargs.get('overrride_token', None))
            else:
                logger.warning(f'learned_embeds_path {learned_embeds_path} does not exist!')
                
        if vram_O:
            # this will change device from gpu to other types (meta)
            enable_vram(pipe)
        else:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
            pipe.to(device)

        self.scheduler = DDIMScheduler.from_pretrained(
            sd_path, subfolder="scheduler", torch_dtype=self.precision_t, local_files_only=False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device)  # for convenience

        logger.info(f'[INFO] loaded stable diffusion!')

        # for CLIP
        self.use_clip = use_clip
        if self.use_clip:
            #breakpoint()
            self.clip_model = CLIPModel.from_pretrained(clip_path).to(device)
            image_processor = CLIPProcessor.from_pretrained(clip_path).image_processor
            self.image_processor = transforms.Compose([
                transforms.Resize((image_processor.crop_size['height'], image_processor.crop_size['width'])),
                transforms.Normalize(image_processor.image_mean, image_processor.image_std),
            ])
            for p in self.clip_model.parameters():
                p.requires_grad = False
                
        self.clip_iterative = clip_iterative
        self.clip_t = int(self.num_train_timesteps *  clip_t)
 
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    @torch.no_grad()
    def get_all_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))
        # text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        # return all text embeddings and class embeddings
        return torch.cat([text_embeddings[0], text_embeddings[1].unsqueeze(1)], dim=1)

    # @torch.no_grad()
    def get_clip_img_embeds(self, img):
        img = self.image_processor(img)
        image_z = self.clip_model.get_image_features(img)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features
        return image_z

    def clip_loss(self, ref_z, pred_rgb):
        image_z = self.get_clip_img_embeds(pred_rgb) 
        loss = spherical_dist_loss(image_z, ref_z)
        return loss
    
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_clip=None, grad_scale=1.0,
                   image_ref_clip=None, text_ref_clip=None, clip_guidance=100, clip_image_loss=False,
                   density=None, 
                   save_guidance_path=None
                   ):
        enable_clip = self.use_clip and clip_guidance > 0 and not as_latent 
        enable_sds = True
        #breakpoint() 
        if as_latent:
            latents = F.interpolate(
                pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # Since during the optimzation, the 3D is getting better.
        # mn = max(self.min_step, int(self.max_step - (self.max_step - self.min_step) / (self.opt.max_epoch // 3) * self.epoch + 0.5))
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)
        if enable_clip and self.clip_iterative:
            if t > self.clip_t:
                enable_clip = False
            else:
                enable_sds = False
             
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # Save input tensors for UNet
            # torch.save(latent_model_input, "train_latent_model_input.pt")
            # torch.save(t, "train_t.pt")
            # torch.save(text_embeddings, "train_text_embeddings.pt")
            tt = torch.cat([t]*2)
            noise_pred = self.unet(latent_model_input, tt,
                                encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        if enable_clip:
            pred_original_sample = (latents_noisy - (1 - self.alphas[t]) ** (0.5) * noise_pred) / self.alphas[t] ** (0.5)
            sample = pred_original_sample
            sample = sample.detach().requires_grad_()
            
            sample = 1 / self.vae.config.scaling_factor * sample
            out_image = self.vae.decode(sample).sample
            out_image = (out_image / 2 + 0.5)#.clamp(0, 1)
            image_embeddings_clip = self.get_clip_img_embeds(out_image)
            ref_clip = image_ref_clip if clip_image_loss else text_ref_clip 
            loss_clip = spherical_dist_loss(image_embeddings_clip, ref_clip).mean() * clip_guidance * 50 # 100
            grad_clipd = - torch.autograd.grad(loss_clip, sample, retain_graph=True)[0]
        else:
            grad_clipd = 0 
        
        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)
        
        if density is not None:
            with torch.no_grad():
                density = F.interpolate(density.detach(), (64, 64), mode='bilinear', align_corners=False)
                ids = torch.nonzero(density.squeeze()) 
                spatial_weight = torch.ones_like(density, device=density.device)
                try:
                    up = ids[:, 0].min()
                    down = ids[:, 0].max() + 1
                    ll = ids[:, 1].min()
                    rr = ids[:, 1].max() + 1
                    spatial_weight[:, :, up:down, ll:rr] += 1
                except:
                    pass
            # breakpoint()
        # w(t), sigma_t^2
        w = (1 - self.alphas[t])[:, None, None, None]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])

        if enable_sds:
            grad_sds = grad_scale * w * (noise_pred - noise)
            loss_sds = grad_sds.abs().mean().detach() 
        else:
            grad_sds = 0. 
            loss_sds = 0.

        if enable_clip:
            grad_clipd = w * grad_clipd.detach()
            loss_clipd = grad_clipd.abs().mean().detach()
        else:
            grad_clipd = 0.  
            loss_clipd = 0.
        
        grad = grad_clipd + grad_sds
        
        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)

        if density is not None:
            grad = grad * spatial_weight / 2

        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # loss = SpecifyGradient.apply(latents, grad)
        # loss = loss.abs().mean().detach()
        latents.backward(gradient=grad, retain_graph=True)
        loss = grad.abs().mean().detach()

        if not enable_clip:
            loss_sds = loss

        if save_guidance_path:
            with torch.no_grad():
                # save original input
                images = []
                os.makedirs(os.path.dirname(save_guidance_path), exist_ok=True)
                timesteps = torch.arange(-1, 1000, 100, dtype=torch.long, device=self.device)
                timesteps[0] *= 0
                for t in timesteps:
                    if as_latent:
                        pred_rgb_512 = self.decode_latents(latents) 
                    latents_noisy = self.scheduler.add_noise(latents, noise, t)
                    
                    # pred noise
                    latent_model_input = torch.cat([latents_noisy] * 2)

                    noise_pred = self.unet(latent_model_input, t,
                                        encoder_hidden_states=text_embeddings).sample

                    # perform guidance (high scale from paper!)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                    pred_original_sample = self.decode_latents((latents_noisy - (1 - self.alphas[t]) ** (0.5) * noise_pred) / self.alphas[t] ** (0.5))

                    # visualize predicted denoised image
                    # claforte: discuss this with Vikram!!
                    result_hopefully_less_noisy_image = self.decode_latents(latents - w*(noise_pred - noise))

                    # visualize noisier image
                    result_noisier_image = self.decode_latents(latents_noisy) 

                    # add in the last col, w/o rendered view contraint, using random noise as latent.
                    latent_model_input = torch.cat([noise] * 2)
                    noise_pred = self.unet(latent_model_input, t,
                            encoder_hidden_states=text_embeddings).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + guidance_scale * \
                            (noise_pred_text - noise_pred_uncond)
                    noise_diffusion_out = self.decode_latents((noise - (1 - self.alphas[t]) ** (0.5) * noise_pred) / self.alphas[t] ** (0.5))
                    # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                    image = torch.cat([pred_rgb_512, pred_original_sample, result_noisier_image, result_hopefully_less_noisy_image, noise_diffusion_out],dim=0)
                    images.append(image)
                viz_images = torch.cat(images, dim=0)
                save_image(viz_images, save_guidance_path, nrow=5)

        return loss, {'loss_sds': loss_sds, 'loss_clipd': loss_clipd} 

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                # latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # Save input tensors for UNet
                # torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
                # torch.save(t, "produce_latents_t.pt")
                # torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents)[
                    'prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        # with torch.no_grad():
        imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def encode_imgs_mean(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        latents = self.vae.encode(imgs).latent_dist.mean
        latents = latents * self.vae.config.scaling_factor

        return latents

    @torch.no_grad()
    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, to_numpy=True):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts) 

        prompts = tuple(prompts)
        negative_prompts = tuple(negative_prompts)
        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts)  # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat(
            [neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents.to(
            text_embeds.dtype))  # [1, 3, 512, 512]

        # Img to Numpy
        if to_numpy:
            imgs = to_np_img(imgs)
        return imgs

    @torch.no_grad()
    def img_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, img=None, to_numpy=True, t=50):
        """
        Known issues:
        1. Not able to reconstruct images even with no noise.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts)  # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat(
            [neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

        # image to latent
        # interp to 512x512 to be fed into vae.
        if isinstance(img, str):
            img = TVF.to_tensor(Image.open(img))[None, :3].cuda()

        img_512 = F.interpolate(
            img.to(text_embeds.dtype), (512, 512), mode='bilinear', align_corners=False)
        # logger.info(img_512.shape, img_512, '\n', img_512.min(), img_512.max(), img_512.mean())

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(img_512).repeat(
            text_embeds.shape[0] // 2, 1, 1, 1)
        # logger.info(latents.shape, latents, '\n', latents.min(), latents.max(), latents.mean())

        noise = torch.randn_like(latents)
        if t > 0:
            latents_noise = self.scheduler.add_noise(
                latents, noise, torch.tensor(t).to(torch.int32))
        else:
            latents_noise = latents

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents_noise,
                                       num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents.to(
            text_embeds.dtype))  # [1, 3, 512, 512]

        # Img to Numpy
        if to_numpy:
            imgs = to_np_img(imgs)
        return imgs

    def add_tokens_to_model(self, learned_embeds: Mapping[str, Tensor], override_token: Optional[Union[str, dict]] = None) -> None:
        r"""Adds tokens to the tokenizer and text encoder of a model."""

        # Loop over learned embeddings
        new_tokens = []
        for token, embedding in learned_embeds.items():
            embedding = embedding.to(
                self.text_encoder.get_input_embeddings().weight.dtype)
            if override_token is not None:
                token = override_token if isinstance(
                    override_token, str) else override_token[token]

            # Add the token to the tokenizer
            num_added_tokens = self.tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError((f"The tokenizer already contains the token {token}. Please pass a "
                                  "different `token` that is not already in the tokenizer."))

            # Resize the token embeddings
            self.text_encoder._resize_token_embeddings(len(self.tokenizer))

            # Get the id for the token and assign the embeds
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.text_encoder.get_input_embeddings(
            ).weight.data[token_id] = embedding
            new_tokens.append(token)

        logger.info(
            f'Added {len(new_tokens)} tokens to tokenizer and text embedding: {new_tokens}')

    def add_tokens_to_model_from_path(self, learned_embeds_path: str, override_token: Optional[Union[str, dict]] = None) -> None:
        r"""Loads tokens from a file and adds them to the tokenizer and text encoder of a model."""
        learned_embeds: Mapping[str, Tensor] = torch.load(
            learned_embeds_path, map_location='cpu')
        self.add_tokens_to_model(learned_embeds, override_token)

    def check_prompt(self, opt):
        texts = ['', ', front view', ', side view', ', back view'] 
        for view_text in texts:
            text = opt.text + view_text
            logger.info(f'Checking stable diffusion model with prompt: {text}')
            # Generate
            image_check = self.prompt_to_img(
                prompts=[text] * opt.get('prompt_check_nums', 5), guidance_scale=7.5, to_numpy=False, 
                num_inference_steps=opt.get('num_inference_steps', 50))
            # Save
            output_dir_check = Path(opt.workspace) / 'prompt_check'
            output_dir_check.mkdir(exist_ok=True, parents=True)
            to_pil(image_check).save(output_dir_check / f'generations_{view_text}.png')
            (output_dir_check / 'prompt.txt').write_text(text)



if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt
    from easydict import EasyDict as edict
    import glob
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--workspace', default='out/sd', type=str)
    parser.add_argument('--image_path', default=None, type=str)
    parser.add_argument('--learned_embeds_path', type=str,
                        default=None, help="path to learned embeds"
                        )
    parser.add_argument('--sd_version', type=str, default='1.5',
                        choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None,
                        help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true',
                        help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true',
                        help="optimization for low VRAM usage")
    parser.add_argument('--gudiance_scale', type=float, default=100)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--noise_t', type=int, default=50)
    parser.add_argument('--prompt_check_nums', type=int, default=5)
    opt, unknown = parser.parse_known_args()

    # seed_everything(opt.seed)
    device = torch.device('cuda')
    opt = edict(vars(opt))
    workspace = opt.workspace

    opt.original_text = opt.text
    opt.original_negative = opt.negative
    if opt.learned_embeds_path is not None: 
        # cml: 
        # python guidance/sd_utils.py --text "A high-resolution DSLR image of <token>" --learned_embeds_path out/learned_embeds/  --workspace out/teddy_bear
        # check prompt
        if os.path.isdir(opt.learned_embeds_path):
            learned_embeds_paths = glob.glob(os.path.join(opt.learned_embeds_path, 'learned_embeds*bin'))
        else:
            learned_embeds_paths = [opt.learned_embeds_path] 
        
        for learned_embeds_path in learned_embeds_paths: 
            embed_name = os.path.basename(learned_embeds_path).split('.')[0]
            opt.workspace = os.path.join(workspace, embed_name) 
            sd = StableDiffusion(device, opt.fp16, opt.vram_O,
                                opt.sd_version, opt.hf_key,
                                learned_embeds_path=learned_embeds_path
                                )
            # Add tokenizer
            if learned_embeds_path is not None:  # add textual inversion tokens to model
                opt.text, opt.negative = token_replace(
                    opt.original_text, opt.original_negative, learned_embeds_path)
                logger.info(opt.text, opt.negative)
                sd.check_prompt(opt)
    else:
        #breakpoint()
        if opt.image_path is not None:
            save_promt = '_'.join(opt.text.split(' ')) + '_' + opt.image_path.split(
                '/')[-1].split('.')[0] + '_' + str(opt.noise_t) + '_' + str(opt.num_inference_steps)
            imgs = sd.img_to_img([opt.text]*opt.prompt_check_nums, [opt.negative]*opt.prompt_check_nums, opt.H, opt.W, opt.num_inference_steps,
                                to_numpy=False, img=opt.image_path, t=opt.noise_t, guidance_scale=opt.gudiance_scale)
        else:
            save_promt = '_'.join(opt.text.split(' '))
            imgs = sd.prompt_to_img([opt.text]*opt.prompt_check_nums, [opt.negative]
                                    * opt.prompt_check_nums, opt.H, opt.W, opt.num_inference_steps, to_numpy=False)
        # visualize image
        output_dir_check = Path(opt.workspace)
        output_dir_check.mkdir(exist_ok=True, parents=True)

        to_pil(imgs).save(output_dir_check / f'{save_promt}.png')
