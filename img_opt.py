import cv2
import imageio
import matplotlib.pyplot as plt
# import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm.auto import tqdm

# from diffusers import StableDiffusionPipeline
from diffusers.models.vae import DiagonalGaussianDistribution
from diffusers.schedulers import LMSDiscreteScheduler

from diffusers import StableDiffusionPipeline

model_id_or_path = "CompVis/stable-diffusion-v1-4"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id_or_path,
    # revision="fp16",
    # torch_dtype=torch.float16,
    # use_auth_token=False
)
pipe = pipe.to("cuda")

def decode_latents(pipe, latents):
    with torch.no_grad():
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        # image = pipe.vae.decode(latents.to(self.vae.dtype)).sample
        image = pipe.vae.decoder(pipe.vae.post_quant_conv(latents.to(pipe.vae.dtype)))
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
    return image


def img_train(latents, optim, pipe, prompt, n_iters, guidance_scale=40.0, sample_freq=10, sorted=True):
    try:
        # expand the latents if we are doing classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        with torch.no_grad():
            # get prompt text embeddings
            text_input = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.text_encoder.device))[0]

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                uncond_input = pipe.tokenizer(
                    "", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
                )
                uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.text_encoder.device))[0]

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        num_train_timesteps = pipe.scheduler.config.num_train_timesteps
        min_step = int(num_train_timesteps * 0.02)
        max_step = int(num_train_timesteps * 0.98)

        text_unet = None
        losses, losses_abs, losses_diff, losses_diff_x0, samples = [], [], [], [], []

        for i in tqdm(range(n_iters)):

            with torch.no_grad():

                if sorted:
                    timesteps = torch.randint(min_step, max_step + 1, (latents.shape[0],), dtype=torch.long, device=pipe.device).sort()[0]
                else:
                    timesteps = torch.randint(min_step, max_step + 1, (latents.shape[0],), dtype=torch.long, device=pipe.device)

                # add noise to latents using the timesteps
                noise = torch.randn(latents.shape, generator=None, device=pipe.device)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps).to(pipe.device)

                # predict the noise residual
                samples_unet = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
                timesteps_unet = torch.cat([timesteps] * 2) if do_classifier_free_guidance else timesteps
                text_unet = text_embeddings.repeat_interleave(len(noisy_latents), 0) if text_unet is None else text_unet
                noise_pred = []
                for s, t, text in zip(samples_unet, timesteps_unet, text_unet):
                    noise_pred.append(pipe.unet(sample=s[None, ...],
                                                timestep=t[None, ...],
                                                encoder_hidden_states=text[None, ...]).sample)

                noise_pred = torch.cat(noise_pred)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            optim.zero_grad()
            loss = ((noise_pred - noise) * latents).mean()
            loss.backward()
            optim.step()

            loss_abs = ((noise_pred - noise) * latents).abs().mean()
            loss_diffusion = ((noise_pred - noise)**2).mean()
            # loss_diffusion_x0 = ((latents - (noisy_latents - torch.sqrt(1 - pipe.scheduler.alphas_cumprod[timesteps])[:, None, None, None].to(latents.device) * noise_pred)/torch.sqrt(pipe.scheduler.alphas_cumprod[timesteps])[:, None, None, None])**2).mean()

            losses.append(loss.item())
            losses_abs.append(loss_abs.item())
            losses_diff.append(loss_diffusion.item())
            # losses_diff_x0.append(loss_diffusion_x0.item())

            if i % sample_freq == 0:
                # print(i, loss.item())
                # media.show_images(latents.detach().cpu().permute(1, 0, 2, 3).reshape(-1, 64, 64), columns=len(latents))
                s = decode_latents(pipe, latents.detach())
                # media.show_images(s, height=100)
                samples.append(s)

    except KeyboardInterrupt:
        print("Ctrl+C!")

    # return latents, [losses, losses_abs, losses_diff, losses_diff_x0], samples
    return latents, [losses, losses_abs, losses_diff], samples


def make_gif_from_imgs(frames, imsize=512, resize=1.0, fps=3, upto=None, repeat_first=1, repeat_last=4, skip=1,
             f=0, s=3, t=2):
    imgs = []
    for i, img in tqdm(enumerate(frames[:upto:skip]), total=len(frames[:upto:skip])):
        img = np.moveaxis(img, 0, 1).reshape(512, -1, 3)
        n_cols = img.shape[1]//imsize
        if resize > 1:
            img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize((int(img.shape[1]/resize), int(img.shape[0]/resize)), Image.Resampling.LANCZOS))
        else:
            img = np.array(Image.fromarray((img*255).astype(np.uint8)))
        # img = np.concatenate([img[:, int(i*imsize/resize):int((i+1)*imsize/resize), :] for i in range(0, n_cols, 2)], axis=1)
        text = f"{i*10:05d}"
        img = cv2.putText(img=img, text=text, org=(0, 20), fontFace=f, fontScale=s/resize, color=(0,0,0), thickness=t)
        imgs.append(img)
    # Save gif
    imgs = [imgs[0]]*repeat_first + imgs + [imgs[-1]]*repeat_last
    return imgs


torch.manual_seed(0)
latents = torch.randn((10, 4, 64, 64), device=pipe.device, requires_grad=True)
optim = torch.optim.Adam([latents], lr=1e-1)
latents, losses, samples = img_train(latents, optim, pipe, "A 3D render of a single strawberry centered", 101, 40.0, sorted=True)

gif = make_gif_from_imgs(samples, resize=2)
# imageio.mimwrite("a.gif", gif, fps=4)
# imageio.mimwrite("a.gif", gif, duration=(len(gif)/4*1000))
imageio.mimwrite("a.mp4", gif, fps=4, quality=8, macro_block_size=1)