import sys
import argparse
import json
import numpy as np
import torch
import gradio as gr
import zipfile

from nerf.provider import NeRFDataset
from nerf.utils import *

import preprocess_image
from guidance import zero123_utils
from torchvision import transforms
from ldm.models.diffusion.ddim import DDIMSampler
from torch import autocast
from contextlib import nullcontext
from einops import rearrange
from omegaconf import OmegaConf

# functions used by backup

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
    # taken from zero123/gradio_new.py
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
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

# functions used by events

def update_workspaces_list():
    workspaces = os.listdir("workspaces/")
    return gr.Dropdown.update(choices=workspaces)

def update_max_epoch_fetch(workspace):
    try:
        items = os.listdir("workspaces/"+workspace+"/results")
        for item in items:
            index = str(item).find("df_ep")+5
            if index > 0:
                return gr.Number.update(value=int(str(item)[index:index+4]))
    except:
        pass        

def update_iters_fetch(workspace):
    try:
        with open("workspaces/"+workspace+"/log_df.txt", "r") as file:
            text = file.read()
            start = text.find(", iters=")+8
            end = text.find(", lr=")
        return gr.Number.update(value=int(text[start:end])*2)
    except:
        print("Invalid log_df file")

def update_max_epoch_calculate(iters, dataset_size_train, batch_size):
    return gr.Number.update(value=int(np.ceil(iters / dataset_size_train*batch_size)))

def update_image_slider(max_epoch, workspace, slider, dmtet):
    if dmtet:
        workspace = str(workspace)+"_dmtet"
    # updates the image based on the slider value, usually to select the "angle" (index of the image)
    image = cv2.imread("temp/"+workspace+"/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, slider))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return gr.Image.update(value=image)

# functions used by gradio ui buttons

def preprocess_images(image):
    image_rgba, image_depth, image_normal = preprocess(image)
    image_normal = cv2.cvtColor(image_normal, cv2.COLOR_RGB2BGR)
    
    return image_rgba, image_depth, image_normal

def generate_novel_view(image, polar, azimuth, radius): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = zero123_utils.load_model_from_config(
        OmegaConf.load("pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"),
        "pretrained/zero123/zero123-xl.ckpt",
        device)
    model.use_ema = False
    
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    image = transforms.functional.resize(image, [256, 256], antialias=True)
    
    sampler = DDIMSampler(model)
    x_samples_ddim = sample_model(image, model, sampler, "fp32", 256, 256, 500, 1, 3.0, 1.0, polar, azimuth, radius)

    novel_image = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        # novel_image.append(Image.fromarray(x_sample.astype(np.uint8)))
        novel_image.append(np.asarray(x_sample.astype(np.uint8)))
        
    torch.cuda.empty_cache()
    
    final_image = preprocess(novel_image[0])[0]
    
    return final_image

def generate_novel_view_radii(image, polar, azimuth):
    radii = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = zero123_utils.load_model_from_config(
        OmegaConf.load("pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"),
        "pretrained/zero123/zero123-xl.ckpt",
        device)
    model.use_ema = False
    
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    image = transforms.functional.resize(image, [256, 256], antialias=True)
    
    sampler = DDIMSampler(model)
    results = []
    for i in range(11):
        x_samples_ddim = sample_model(image, model, sampler, "fp32", 256, 256, 500, 1, 3.0, 1.0, polar, azimuth, radii[i])
        novel_image = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            # novel_image.append(Image.fromarray(x_sample.astype(np.uint8)))
            novel_image.append(np.asarray(x_sample.astype(np.uint8)))
        results.append(novel_image[0])
        del x_samples_ddim
        del novel_image
    
    torch.cuda.empty_cache()
    
    return results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9], results[10]

def generate_novel_view_multi(image, radius):
    polars = [0.0, 0.0, 0.0, 0.0, -90.0, 90.0]
    azimuths = [0.0, 90.0, 180.0, -90.0, 0.0, 0.0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = zero123_utils.load_model_from_config(
        OmegaConf.load("pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"),
        "pretrained/zero123/zero123-xl.ckpt",
        device)
    model.use_ema = False
    
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    image = transforms.functional.resize(image, [256, 256], antialias=True)
    
    sampler = DDIMSampler(model)
    results = []
    for i in range(6):
        x_samples_ddim = sample_model(image, model, sampler, "fp32", 256, 256, 500, 1, 3.0, 1.0, polars[i], azimuths[i], radius)
        novel_image = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            # novel_image.append(Image.fromarray(x_sample.astype(np.uint8)))
            novel_image.append(np.asarray(x_sample.astype(np.uint8)))
        # results.append(preprocess(novel_image[0])[0])
        results.append(novel_image[0])
        del x_samples_ddim
        del novel_image
    
    torch.cuda.empty_cache()
    
    return gr.Textbox.update(visible=True), gr.Number.update(visible=True), gr.Number.update(visible=True), gr.Number.update(visible=True), gr.Number.update(visible=True), gr.Number.update(visible=True), \
           gr.Number.update(visible=True), gr.Number.update(visible=True), gr.Number.update(visible=True), gr.Number.update(visible=True), gr.Number.update(visible=True), gr.Number.update(visible=True), \
           gr.Button.update(visible=True), gr.Image.update(value=results[0], visible=True), gr.Image.update(value=results[1], visible=True), gr.Image.update(value=results[2], visible=True), \
           gr.Image.update(value=results[3], visible=True), gr.Image.update(value=results[4], visible=True), gr.Image.update(value=results[5], visible=True)

def generate_3d_model(image, workspace, seed, size, iters, lr, batch_size, dataset_size_train, dataset_size_valid, dataset_size_test, backbone, optim, fp16, max_epoch):
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
    # these arguments are for zero123
    opt["backbone"] = backbone
    opt["optim"] = optim
    opt["fp16"] = fp16
    opt["workspace"] = "workspaces/"+workspace
    opt["seed"] = seed
    opt["h"] = size
    opt["w"] = size
    opt["iters"] = iters
    opt["lr"] = lr
    opt["batch_size"] = batch_size
    opt["dataset_size_train"] = dataset_size_train
    opt["dataset_size_valid"] = dataset_size_valid
    opt["dataset_size_test"] = dataset_size_test
    opt["images"] = ["temp/"+workspace+"/image_rgba.png"]
    opt["exp_start_iter"] = opt["exp_start_iter"] or 0
    opt["exp_end_iter"] = opt["exp_end_iter"] or opt["iters"]
    
    opt = argparse.Namespace(**opt)
    
    print(opt)
    
    # preprocess and save images to temporary folder in order to make calls to stable-dreamfusion trainer without making changes to its code
    image_rgba, image_depth, image_normal = preprocess(image, size=1024)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("temp/"+workspace, exist_ok=True)
    cv2.imwrite("temp/"+workspace+"/image_rgba.png", cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite("temp/"+workspace+"/image_depth.png", image_depth)
    cv2.imwrite("temp/"+workspace+"/image_normal.png", image_normal)
    del image_rgba, image_depth, image_normal
    
    # this part of the code loads up the relevant modules to allow the trainer to run (taken from main.py)    
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRFNetwork(opt).to(device)

    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

    if opt.optim == 'adan':
        from optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    else:
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    guidance = nn.ModuleDict()
    guidance['zero123'] = zero123_utils.Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)
    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)
    trainer.default_view_data = train_loader._data.get_default_view_data()
    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
    test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

    # starts the training
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)
    
    video = cv2.VideoCapture("workspaces/"+workspace+"/results/df_ep{:04d}_rgb.mp4".format(max_epoch))
    for i in range(100):
        image = video.read()[1]
        cv2.imwrite("temp/"+workspace+"/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, i), image)
    
    # saves mesh
    trainer.save_mesh()
    with zipfile.ZipFile("temp/"+workspace+"/"+workspace+".zip", "w") as file:
        file.write("workspaces/"+workspace+"/mesh/albedo.png", arcname="albedo.png")
        file.write("workspaces/"+workspace+"/mesh/mesh.mtl", arcname="mesh.mtl")
        file.write("workspaces/"+workspace+"/mesh/mesh.obj", arcname="mesh.obj")
    
    torch.cuda.empty_cache()
    
    return gr.Slider.update(maximum=99, value=0), gr.File.update(value="temp/"+workspace+"/"+workspace+".zip", visible=True)

def generate_3d_model_multi(image1, image2, image3, image4, image5, image6, radius, workspace, seed, size, iters, lr, batch_size, dataset_size_train, dataset_size_valid, dataset_size_test, backbone, optim, fp16, max_epoch):
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
    # these arguments are for zero123
    opt["backbone"] = backbone
    opt["optim"] = optim
    opt["fp16"] = fp16
    opt["workspace"] = "workspaces/"+workspace
    opt["seed"] = seed
    opt["h"] = size
    opt["w"] = size
    opt["iters"] = iters
    opt["lr"] = lr
    opt["batch_size"] = batch_size
    opt["dataset_size_train"] = dataset_size_train
    opt["dataset_size_valid"] = dataset_size_valid
    opt["dataset_size_test"] = dataset_size_test
    opt["images"] = ["temp/"+workspace+"/image1_rgba.png", "temp/"+workspace+"/image2_rgba.png", "temp/"+workspace+"/image3_rgba.png", "temp/"+workspace+"/image4_rgba.png", "temp/"+workspace+"/image5_rgba.png", "temp/"+workspace+"/image6_rgba.png"]
    opt["ref_radii"] = [radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001]
    opt["ref_polars"] = [90.0, 90.0, 90.0, 90.0, 180.0, 0.0001]
    opt["ref_azimuths"] = [0.0, 90.0, 180.0, -90.0, 0.0, 0.0]
    opt["zero123_ws"] = [5, 5, 5, 5, 1, 1] #[0.225, 0.225, 0.225, 0.225, 0.05, 0.05]
    opt["exp_start_iter"] = opt["exp_start_iter"] or 0
    opt["exp_end_iter"] = opt["exp_end_iter"] or opt["iters"]
    
    opt = argparse.Namespace(**opt)
    
    print(opt)
    
    # preprocess and save images to temporary folder in order to make calls to stable-dreamfusion trainer without making changes to its code
    images = [image1, image2, image3, image4, image5, image6]
    for i in range(1, 7):
        image_rgba, image_depth, image_normal = preprocess(images[i-1], size=1024)
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/"+workspace, exist_ok=True)
        cv2.imwrite("temp/"+workspace+"/image"+str(i)+"_rgba.png", cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite("temp/"+workspace+"/image"+str(i)+"_depth.png", image_depth)
        cv2.imwrite("temp/"+workspace+"/image"+str(i)+"_normal.png", image_normal)
        del image_rgba, image_depth, image_normal
    
    # this part of the code loads up the relevant modules to allow the trainer to run (taken from main.py)    
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRFNetwork(opt).to(device)

    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

    if opt.optim == 'adan':
        from optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    else:
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    guidance = nn.ModuleDict()
    guidance['zero123'] = zero123_utils.Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)
    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)
    trainer.default_view_data = train_loader._data.get_default_view_data()
    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
    test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

    # starts the training
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)
    
    video = cv2.VideoCapture("workspaces/"+workspace+"/results/df_ep{:04d}_rgb.mp4".format(max_epoch))
    for i in range(100):
        image = video.read()[1]
        cv2.imwrite("temp/"+workspace+"/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, i), image)
    
    # saves mesh
    trainer.save_mesh()
    with zipfile.ZipFile("temp/"+workspace+"/"+workspace+".zip", "w") as file:
        file.write("workspaces/"+workspace+"/mesh/albedo.png", arcname="albedo.png")
        file.write("workspaces/"+workspace+"/mesh/mesh.mtl", arcname="mesh.mtl")
        file.write("workspaces/"+workspace+"/mesh/mesh.obj", arcname="mesh.obj")
    
    torch.cuda.empty_cache()
    
    return gr.Image.update(visible=True), gr.Slider.update(maximum=99, value=0, visible=True), gr.File.update(value="temp/"+workspace+"/"+workspace+".zip", visible=True)


def finetune_3d_model(workspace, seed, tet_grid_size, iters, lr, backbone, optim, fp16, max_epoch):
    if seed == 0:
        seed = None
        
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
    # these arguments are for zero123
    opt["backbone"] = backbone
    opt["optim"] = optim
    opt["fp16"] = fp16
    opt["workspace"] = "workspaces/"+workspace+"_dmtet"
    opt["seed"] = seed
    opt["iters"] = iters
    opt["lr"] = lr
    opt["dmtet"] = True
    opt["init_with"] = "workspaces/"+workspace+"/checkpoints/df.pth"
    opt["tet_grid_size"] = int(tet_grid_size)
    opt["images"] = ["temp/"+workspace+"_dmtet/image_rgba.png"]
    opt["exp_start_iter"] = opt["exp_start_iter"] or 0
    opt["exp_end_iter"] = opt["exp_end_iter"] or opt["iters"]
    opt.pop("full_radius_range")
    opt.pop("full_theta_range")
    opt.pop("full_phi_range")
    opt.pop("full_fovy_range")
    
    opt = argparse.Namespace(**opt)
    
    print(opt)
    
    image = cv2.imread("workspaces/"+workspace+"/image_rgba.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image_rgba, image_depth, image_normal = preprocess(image, size=1024)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("temp/"+workspace+"_dmtet", exist_ok=True)
    cv2.imwrite("temp/"+workspace+"_dmtet/image_rgba.png", cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite("temp/"+workspace+"_dmtet/image_depth.png", image_depth)
    cv2.imwrite("temp/"+workspace+"_dmtet/image_normal.png", image_normal)
    del image_rgba, image_depth, image_normal
    
    opt.h = int(opt.h * opt.dmtet_reso_scale)
    opt.w = int(opt.w * opt.dmtet_reso_scale)
    opt.known_view_scale = 1
    if not opt.dont_override_stuff:            
        opt.t_range = [0.02, 0.50] # ref: magic3D
    if opt.images is not None:
        opt.lambda_normal = 0
        opt.lambda_depth = 0
        if opt.text is not None and not opt.dont_override_stuff:
            opt.t_range = [0.20, 0.50]
    # assume finetuning
    opt.latent_iter_ratio = 0
    opt.albedo_iter_ratio = 0
    opt.progressive_view = False
    # opt.progressive_level = False
    
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)
        
    print(opt)
    
    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRFNetwork(opt).to(device)
    
    if opt.init_with.endswith('.pth'):
        # load pretrained weights to init dmtet
        state_dict = torch.load(opt.init_with, map_location=device)
        model.load_state_dict(state_dict['model'], strict=False)
        if opt.cuda_ray:
            model.mean_density = state_dict['mean_density']
        model.init_tet()
    else:
        # assume a mesh to init dmtet (experimental, not working well now!)
        import trimesh
        mesh = trimesh.load(opt.init_with, force='mesh', skip_material=True, process=False)
        model.init_tet(mesh=mesh)
    
    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

    if opt.optim == 'adan':
        from optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    else:
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    guidance = nn.ModuleDict()
    guidance['zero123'] = zero123_utils.Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)
    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)
    trainer.default_view_data = train_loader._data.get_default_view_data()
    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
    test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

    # starts the training
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)
    
    video = cv2.VideoCapture("workspaces/"+workspace+"_dmtet/results/df_ep{:04d}_rgb.mp4".format(max_epoch))
    for i in range(100):
        image = video.read()[1]
        cv2.imwrite("temp/"+workspace+"_dmtet/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, i), image)
    
    # saves mesh
    trainer.save_mesh()
    with zipfile.ZipFile("temp/"+workspace+"_dmtet/"+workspace+".zip", "w") as file:
        file.write("workspaces/"+workspace+"_dmtet/mesh/albedo.png", arcname="albedo.png")
        file.write("workspaces/"+workspace+"_dmtet/mesh/mesh.mtl", arcname="mesh.mtl")
        file.write("workspaces/"+workspace+"_dmtet/mesh/mesh.obj", arcname="mesh.obj")
    
    return gr.Slider.update(maximum=99, value=0), gr.File.update(value="temp/"+workspace+"_dmtet/"+workspace+".zip", visible=True)

def finetune_3d_model_multi(workspace, seed, tet_grid_size, iters, lr, backbone, optim, fp16, max_epoch):
    if seed == 0:
        seed = None
        
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
    # these arguments are for zero123
    opt["backbone"] = backbone
    opt["optim"] = optim
    opt["fp16"] = fp16
    opt["workspace"] = "workspaces/"+workspace+"_dmtet"
    opt["seed"] = seed
    opt["iters"] = iters
    opt["lr"] = lr
    opt["dmtet"] = True
    opt["init_with"] = "workspaces/"+workspace+"/checkpoints/df.pth"
    opt["tet_grid_size"] = int(tet_grid_size)
    opt["images"] = ["temp/"+workspace+"_dmtet/image1_rgba.png", "temp/"+workspace+"_dmtet/image2_rgba.png", "temp/"+workspace+"_dmtet/image3_rgba.png", "temp/"+workspace+"_dmtet/image4_rgba.png", "temp/"+workspace+"_dmtet/image5_rgba.png", "temp/"+workspace+"_dmtet/image6_rgba.png"]
    opt["ref_radii"] = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001] #[radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001]
    opt["ref_polars"] = [90.0, 90.0, 90.0, 90.0, 180.0, 0.0001]
    opt["ref_azimuths"] = [0.0, 90.0, 180.0, -90.0, 0.0, 0.0]
    opt["zero123_ws"] = [5, 5, 5, 5, 1, 1] #[0.225, 0.225, 0.225, 0.225, 0.05, 0.05]
    opt["exp_start_iter"] = opt["exp_start_iter"] or 0
    opt["exp_end_iter"] = opt["exp_end_iter"] or opt["iters"]
    opt.pop("full_radius_range")
    opt.pop("full_theta_range")
    opt.pop("full_phi_range")
    opt.pop("full_fovy_range")
    
    opt = argparse.Namespace(**opt)
    
    print(opt)
    
    
    for i in range(1, 7):
        image = cv2.imread("workspaces/"+workspace+"/image"+str(i)+"_rgba.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image_rgba, image_depth, image_normal = preprocess(image, size=1024)
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/"+workspace+"_dmtet", exist_ok=True)
        cv2.imwrite("temp/"+workspace+"_dmtet/image"+str(i)+"_rgba.png", cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite("temp/"+workspace+"_dmtet/image"+str(i)+"_depth.png", image_depth)
        cv2.imwrite("temp/"+workspace+"_dmtet/image"+str(i)+"_normal.png", image_normal)
        del image_rgba, image_depth, image_normal
    
    opt.h = int(opt.h * opt.dmtet_reso_scale)
    opt.w = int(opt.w * opt.dmtet_reso_scale)
    opt.known_view_scale = 1
    if not opt.dont_override_stuff:            
        opt.t_range = [0.02, 0.50] # ref: magic3D
    if opt.images is not None:
        opt.lambda_normal = 0
        opt.lambda_depth = 0
        if opt.text is not None and not opt.dont_override_stuff:
            opt.t_range = [0.20, 0.50]
    # assume finetuning
    opt.latent_iter_ratio = 0
    opt.albedo_iter_ratio = 0
    opt.progressive_view = False
    # opt.progressive_level = False
    
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)
        
    print(opt)
    
    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRFNetwork(opt).to(device)
    
    if opt.init_with.endswith('.pth'):
        # load pretrained weights to init dmtet
        state_dict = torch.load(opt.init_with, map_location=device)
        model.load_state_dict(state_dict['model'], strict=False)
        if opt.cuda_ray:
            model.mean_density = state_dict['mean_density']
        model.init_tet()
    else:
        # assume a mesh to init dmtet (experimental, not working well now!)
        import trimesh
        mesh = trimesh.load(opt.init_with, force='mesh', skip_material=True, process=False)
        model.init_tet(mesh=mesh)
    
    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

    if opt.optim == 'adan':
        from optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    else:
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    guidance = nn.ModuleDict()
    guidance['zero123'] = zero123_utils.Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)
    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)
    trainer.default_view_data = train_loader._data.get_default_view_data()
    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
    test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

    # starts the training
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)
    
    video = cv2.VideoCapture("workspaces/"+workspace+"_dmtet/results/df_ep{:04d}_rgb.mp4".format(max_epoch))
    for i in range(100):
        image = video.read()[1]
        cv2.imwrite("temp/"+workspace+"_dmtet/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, i), image)
    
    # saves mesh
    trainer.save_mesh()
    with zipfile.ZipFile("temp/"+workspace+"_dmtet/"+workspace+".zip", "w") as file:
        file.write("workspaces/"+workspace+"_dmtet/mesh/albedo.png", arcname="albedo.png")
        file.write("workspaces/"+workspace+"_dmtet/mesh/mesh.mtl", arcname="mesh.mtl")
        file.write("workspaces/"+workspace+"_dmtet/mesh/mesh.obj", arcname="mesh.obj")
    
    return gr.Slider.update(maximum=99, value=0), gr.File.update(value="temp/"+workspace+"_dmtet/"+workspace+".zip", visible=True)


def load_3d_model(max_epoch, workspace):
    items = os.listdir("workspaces/"+workspace+"/results/")
    
    if not items:
        return gr.Slider.update(maximum=0, value=0)
    for item in items:
        index = str(item).find("df_ep")+5
        if index > 0:
            max_epoch = int(str(item)[index:index+4])
            break
    
    os.makedirs("temp/"+workspace, exist_ok=True)
    video = cv2.VideoCapture("workspaces/"+workspace+"/results/df_ep{:04d}_rgb.mp4".format(max_epoch))
    image = video.read()[1]
    
    cv2.imwrite("temp/"+workspace+"/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, 0), image)
    for i in range(1, 100):
        temp = video.read()[1]
        cv2.imwrite("temp/"+workspace+"/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, i), temp)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    with zipfile.ZipFile("temp/"+workspace+"/"+workspace+".zip", "w") as file:
        file.write("workspaces/"+workspace+"/mesh/albedo.png", arcname="albedo.png")
        file.write("workspaces/"+workspace+"/mesh/mesh.mtl", arcname="mesh.mtl")
        file.write("workspaces/"+workspace+"/mesh/mesh.obj", arcname="mesh.obj")
    
    return max_epoch, image, gr.Slider.update(maximum=99, value=0), gr.File.update(value="temp/"+workspace+"/"+workspace+".zip", visible=True)

# def update_slider_maximum(workspace_input):
#     result = 0
#     try:
#         with open("workspaces/"+workspace_input+"/log_df.txt", "r") as file:
#             temp = file.read()
#             start = temp.find("dataset_size_valid=")+19
#             end = temp.find(", dataset_size_test=")
#             result = int(temp[start:end])
#     except:
#         return gr.Slider.update(maximum=1, value=1)
#     return gr.Slider.update(maximum=result, value=result)