import sys
import argparse
import json
import numpy as np
import torch
import gradio as gr

from nerf.provider import NeRFDataset
from nerf.utils import *
from nerf.network_grid import NeRFNetwork

import preprocess_image
from guidance import zero123_utils
from torchvision import transforms
from ldm.models.diffusion.ddim import DDIMSampler
from torch import autocast
from contextlib import nullcontext
from einops import rearrange
from omegaconf import OmegaConf
from optimizer import Adan

def preprocess(image, recenter=True, size=256, border_ratio=0.2):
    # this checks to if original image is RGBA or RGB then convert it into CV2 format
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # removes the background, keeps only the subject
    carved_image = preprocess_image.BackgroundRemoval()(image) # [H, W, 4]
    mask = carved_image[..., -1] > 0
    
    # predict depth
    dpt_depth_model = preprocess_image.DPT(task='depth')
    depth = dpt_depth_model(image)[0]
    depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
    depth[~mask] = 0
    depth = (depth * 255).astype(np.uint8)
    del dpt_depth_model
    
    # predict normal
    dpt_normal_model = preprocess_image.DPT(task='normal')
    normal = dpt_normal_model(image)[0]
    normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
    normal[~mask] = 0
    del dpt_normal_model
    
    # recenter
    if recenter:
        final_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        final_depth = np.zeros((size, size), dtype=np.uint8)
        final_normal = np.zeros((size, size, 3), dtype=np.uint8)
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    else:
        final_rgba = carved_image
        final_depth = depth
        final_normal = normal
    
    # final_normal = cv2.cvtColor(final_normal, cv2.COLOR_RGB2BGR) # this is only for display onto gradio
    
    return final_rgba, final_depth, final_normal

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def generate_novel_view(image, polar, azimuth, radius):
    polar = float(polar)
    azimuth = float(azimuth)
    radius = float(radius)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = zero123_utils.load_model_from_config(
        OmegaConf.load("./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"),
        "./pretrained/zero123/zero123-xl.ckpt",
        device)
    model.use_ema = False
    
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    image = transforms.functional.resize(image, [256, 256], antialias=True)
    
    sampler = DDIMSampler(model)
    used_polar = polar
    x_samples_ddim = sample_model(image, model, sampler, "fp32", 256, 256, 250, 1, 3.0, 1.0, used_polar, azimuth, radius)

    novel_image = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        #novel_image.append(Image.fromarray(x_sample.astype(np.uint8)))
        novel_image.append(np.asarray(x_sample.astype(np.uint8)))
        
    final_image = preprocess(novel_image[0])[0]
    
    return final_image

def generate_3d_model(image, workspace, size, iters):
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
    # these arguments are for zero123
    opt["workspace"] = "workspaces/"+workspace
    opt["w"] = int(size)
    opt["h"] = int(size)
    opt["images"] = ["./temp/"+workspace+"/image_rgba.png"]
    opt["iters"] = int(iters)
    opt = argparse.Namespace(**opt)
    
    # preprocess and save images to temporary folder in order to make calls to stable-dreamfusion trainer without making changes to its code
    image_rgba, image_depth, image_normal = preprocess(image, size=1024)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("temp/"+workspace, exist_ok=True)
    cv2.imwrite("./temp/"+workspace+"/image_rgba.png", cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite("./temp/"+workspace+"/image_depth.png", image_depth)
    cv2.imwrite("./temp/"+workspace+"/image_normal.png", image_normal)
    del image_rgba, image_depth, image_normal

    # this part of the code loads up the relevant modules to allow the trainer to run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRFNetwork(opt).to(device)
    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()
    optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)
    guidance = nn.ModuleDict()
    guidance['zero123'] = zero123_utils.Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)
    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)
    trainer.default_view_data = train_loader._data.get_default_view_data()
    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
    test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

    # starts the training, save mesh and images
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)
    trainer.save_mesh()

    # remove workspace temp folder
    os.remove("temp/"+workspace+"/image_rgba.png")
    os.remove("temp/"+workspace+"/image_depth.png")
    os.remove("temp/"+workspace+"/image_normal.png")
    os.rmdir("temp/"+workspace)
    
    return "Success."

def preprocess_images(image):
    image_rgba, image_depth, image_normal = preprocess(image)
    image_normal = cv2.cvtColor(image_normal, cv2.COLOR_RGB2BGR)
    
    return image_rgba, image_depth, image_normal

if __name__ == '__main__':
    with gr.Blocks() as demo:
        demo.title = "Zero123D Reconstruction"
        gr.Markdown(
            """
            # Image to 3D Model Generation
            FYP by Rod Oh Zhi Hua.
            """)

        with gr.Tab("Preprocess Image"):
            with gr.Row(equal_height=True):            
                image_input = gr.Image()
                rgba_output = gr.Image()
            with gr.Row(equal_height=True):     
                depth_output = gr.Image()
                normal_output = gr.Image()
            generate_button = gr.Button("Generate")
            generate_button.click(
                fn=preprocess_images,
                inputs=image_input,
                outputs=[rgba_output, depth_output, normal_output])

        with gr.Tab("Generate Novel Views"):
            with gr.Row(equal_height=True):            
                image_input = gr.Image()
                image_output = gr.Image()
            polar_input = gr.Textbox(label="polar", value=0.0) # 90.0
            azimuth_input = gr.Textbox(label="azimuth", value=30.0) # 0.0
            radius_input = gr.Textbox(label="radius", value=0.0) # 3.2
            generate_button = gr.Button("Generate")
            generate_button.click(
                fn=generate_novel_view,
                inputs=[image_input, polar_input, azimuth_input, radius_input],
                outputs=image_output)
            
        with gr.Tab("Generate 3D Model"):
            with gr.Row(equal_height=True):
                image_input = gr.Image()
                image_output = gr.Image()
            workspace_input = gr.Textbox(label="workspace name", value="trial_image")
            size_input = gr.Textbox(label="size (n^2)", value=64) #64
            iters_input = gr.Textbox(label="iters", value=5000) #5000
            text_output = gr.Textbox()
            generate_button = gr.Button("Generate")
            generate_button.click(
                fn=generate_3d_model,
                inputs=[image_input, workspace_input, size_input, iters_input],
                outputs=text_output)

    demo.launch()