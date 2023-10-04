import os
import json
import gradio as gr
import sys
import gc
import torch
import cv2
import zipfile
import preprocess_image
import numpy as np
from contextlib import nullcontext
from nerf.utils import *
from guidance import zero123_utils
from omegaconf import OmegaConf
from torchvision import transforms
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange
import argparse
from nerf.provider import NeRFDataset

# main

def main():
    # global variables used to parse variables between functions that are not users' direct input
    global settings
    global current_tab
    global project_name
    global radius
    global max_epoch

    with gr.Blocks(title="zero123d reconstruction") as app:
        # make temp directory
        os.makedirs("temp", exist_ok=True)
        os.makedirs("workspaces", exist_ok=True)
        
        # initialize global variables and configure gui based on loaded settings
        settings = load_settings()
        current_tab = 3
        if settings["info_tab_on_launch"]:
            current_tab = 0
        max_epoch = 0
        project_name = "temp"
        radius = 0.0

        with gr.Tabs(selected=current_tab) as tabs:
            with gr.Tab(label="new project", id=3) as new_project_tab:
                # components
                project_name_input = gr.Textbox(label="project name (no special characters including spaces, only underscores)")
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, label="image")
                preprocess_image_button = gr.Button(value="preprocess image", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="secondary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                new_project_tab.select(fn=lambda: globals().update(current_tab=3))
                preprocess_image_button.click(fn=lambda: print(end="")).success(fn=preprocess_image_button_handler, inputs=[project_name_input, image_input], outputs=image_input)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
                
            with gr.Tab(label="radius discovery", id=4) as radius_discovery_tab:
                # components
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, interactive=False)
                    images_viewer_output = gr.Image(label="image")
                images_viewer_slider_input = gr.Slider(minimum=0.0, maximum=3.6, label="slide to view images generated with different input radius", step=0.4, interactive=False)
                generate_button = gr.Button(value="generate", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                radius_discovery_tab.select(fn=lambda: globals().update(current_tab=4)).success(fn=return_input_image_handler, outputs=image_input)
                generate_button.click(fn=lambda: print(end="")).success(fn=radius_discovery_handler, outputs=[images_viewer_output, images_viewer_slider_input])
                images_viewer_slider_input.change(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
                
            with gr.Tab(label="six-view generation", id=5) as six_view_generation_tab:
                # components
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, interactive=False)
                    images_viewer_output = gr.Image(label="image")
                images_viewer_slider_input = gr.Slider(minimum=0, maximum=5, label="slide to view images generated from different angles", step=1, interactive=False)
                generate_button = gr.Button(value="generate", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                six_view_generation_tab.select(fn=lambda: globals().update(current_tab=5)).success(fn=return_input_image_handler, outputs=image_input)
                generate_button.click(fn=lambda: print(end="")).success(fn=six_view_generation_handler, outputs=[images_viewer_output, images_viewer_slider_input])
                images_viewer_slider_input.change(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
            
            with gr.Tab(label="model reconstruction", id=6) as model_reconstruction_tab:
                # components
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, interactive=False)
                    images_viewer_output = gr.Image(label="image")
                images_viewer_slider_input = gr.Slider(minimum=0, maximum=99, label="slide to view generated model from different angles", step=1, interactive=False)
                with gr.Row():
                    seed_input = gr.Number(value=None, label="seed", precision=0)
                    size_input = gr.Number(value=64, label="size (n^2, 64 really recommended.)", minimum=64, precision=0, step=1) #64
                with gr.Row():
                    iters_input = gr.Number(value=5000, label="iters (iterations)", precision=0, minimum=1, step=1) #5000
                    lr_input = gr.Number(value=1e-3, label="lr (learning rate)", minimum=1e-5) #1e-3
                    batch_size_input = gr.Number(value=1, label="batch_size", precision=0, minimum=1, step=1) #1
                with gr.Row():
                    dataset_size_train_input = gr.Number(value=100, label="dataset_size_train", precision=0, minimum=1, step=1) #100
                    dataset_size_valid_input = gr.Number(value=8, label="dataset_size_valid", precision=0, minimum=1, step=1) #8
                    dataset_size_test_input = gr.Number(value=100, label="dataset_size_test", precision=0, minimum=1, step=1) #100
                file_output = gr.File(visible=False)
                generate_button = gr.Button(value="generate", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                model_reconstruction_tab.select(fn=lambda: globals().update(current_tab=6)).success(fn=return_input_image_handler, outputs=image_input)
                generate_button.click(fn=lambda: print(end="")).success(fn=model_reconstruction_handler, inputs=[seed_input, size_input, iters_input, lr_input, batch_size_input, dataset_size_train_input, dataset_size_valid_input, dataset_size_test_input], outputs=[images_viewer_output, images_viewer_slider_input, file_output])
                images_viewer_slider_input.change(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
            
            with gr.Tab(label="model fine-tuning", id=7) as model_fine_tuning_tab:
                # components
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, interactive=False)
                    images_viewer_output = gr.Image(label="image")
                images_viewer_slider_input = gr.Slider(minimum=0, maximum=99, label="slide to view generated model from different angles", step=1, interactive=False)
                with gr.Row():
                    seed_input = gr.Number(value=None, label="seed", precision=0)
                    size_input = gr.Number(value=64, label="size (n^2, 64 really recommended.)", minimum=64, precision=0, step=1) #64
                    tet_grid_size_input = gr.Dropdown(label="tet_grid_size", choices=["32", "64", "128", "256"], value="128")
                with gr.Row():
                    iters_input = gr.Number(value=5000, label="iters (iterations)", precision=0, minimum=1, step=1) #5000
                    lr_input = gr.Number(value=1e-3, label="lr (learning rate)", minimum=1e-5) #1e-3
                    batch_size_input = gr.Number(value=1, label="batch_size", precision=0, minimum=1, step=1) #1
                with gr.Row():
                    dataset_size_train_input = gr.Number(value=100, label="dataset_size_train", precision=0, minimum=1, step=1) #100
                    dataset_size_valid_input = gr.Number(value=8, label="dataset_size_valid", precision=0, minimum=1, step=1) #8
                    dataset_size_test_input = gr.Number(value=100, label="dataset_size_test", precision=0, minimum=1, step=1) #100
                file_output = gr.File(visible=False)
                generate_button = gr.Button(value="generate", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="secondary")
                # events
                model_fine_tuning_tab.select(fn=lambda: globals().update(current_tab=7)).success(fn=return_input_image_handler, outputs=image_input)
                generate_button.click(fn=lambda: print(end="")).success(fn=model_finetuning_handler, inputs=[seed_input, size_input, tet_grid_size_input, iters_input, lr_input, batch_size_input, dataset_size_train_input, dataset_size_valid_input, dataset_size_test_input], outputs=[images_viewer_output, images_viewer_slider_input, file_output])
                images_viewer_slider_input.change(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
            
            with gr.Tab(label="project manager", id=2) as file_manager_tab:
                # components
                gr.Markdown(
                    """
                    ### this tab is used to view, download and delete projects, simply select a project name to start.
                    """
                )
                with gr.Row():
                    model_viewer_output = gr.Image(label="model viewer", interactive=False)
                    with gr.Column():
                        model_viewer_slider_input = gr.Slider(minimum=0, maximum=0, label="slide to change viewpoint of model", step=1)
                        project_name_input = gr.Dropdown(choices=os.listdir("workspaces"), label="project name")
                        delete_button = gr.Button(visible=False)
                        file_output = gr.File(visible=False)
                # events
                file_manager_tab.select(fn=lambda: globals().update(current_tab=2))
                model_viewer_slider_input.change(fn=images_viewer_slider_project_manager_handler, inputs=[project_name_input, model_viewer_slider_input], outputs=model_viewer_output)
                project_name_input.input(fn=load_project, inputs=project_name_input, outputs=[model_viewer_output, model_viewer_slider_input, delete_button, file_output])
                delete_button.click(fn=delete_project, inputs=project_name_input, outputs=[model_viewer_output, model_viewer_slider_input, project_name_input, delete_button, file_output])
                
            with gr.Tab(label="settings", id=1) as settings_tab:
                # components
                info_tab_on_launch = gr.Checkbox(value=settings["info_tab_on_launch"], label="load up info tab on launch")
                backbone_input = gr.Dropdown(choices=["grid", "vanilla", "grid_tcnn", "grid_taichi"], value="grid", label="nerf backbone")
                optimizer_input = gr.Dropdown(choices=["adan", "adam"], value="adan", label="optimizer")
                fp16_input = gr.Checkbox(label="use float16 instead of float32 for training ", value=True)
                settings_tab_save_button = gr.Button(value="save settings", variant="primary")
                # events
                settings_tab.select(fn=lambda: globals().update(current_tab=1))
                settings_tab_save_button.click(fn=save_settings, inputs=[info_tab_on_launch, backbone_input, optimizer_input, fp16_input])
                    
            with gr.Tab(label="info", id=0) as info_tab:
                # components
                gr.Markdown(
                    """
                    # image to 3d model generation
                    a final year project by oh zhi hua (rod) for nanyang technological university computer engineering program.
                    
                    ## Introduction
                    This project provides a graphical user interface to generate 3D models from a single image by wrapping the stable-dreamfusion with gradio.
                    
                    As the quality of the 3D model depends largely on the quality of the image generated by stable-diffusion, any unsatisfactory image will ruin the end result.
                    
                    Therefore, the project also provide a way to generate novel viewpoints of the object in the input image, which is then fed into stable-dreamfusion for 3D reconstruction.
                    
                    To start, simply click the tab labeled "reconstruction" to start exploring.
                    
                    Have fun!
                    
                    ## Tabs
                    rod's workflow ==> generate 3D models from a single image using rod's workflow
                    
                    file manager   ==> manage the existing projects, removing (deleting) unwanted projects, cleaning temp files
                    
                    settings       ==> settings page to configure the default values when starting the application
                    
                    ## Support
                    If you need support, please submit an issue at "https://github.com/ghotinggoad/stable-dreamfusion-gui/issues"
                    I will check if the bug is from my wrapper or from stable-dreamfusion!
                    """)
                # events
                info_tab.select(fn=lambda: globals().update(current_tab=0))
    
    app.queue(max_size=1).launch(quiet=True)
    
    # rmdir temp folder
    # delete_directory("temp")
    # clear ram (including vram)
    clear_memory()

# functions used to interact with system

def clear_memory():
    # called after deleting the items in python
    gc.collect()
    torch.cuda.empty_cache()

def delete_directory(path):
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(path)
    except:
        print("cannot delete the folder "+path)

def load_settings():
    try:
        with open("settings.json", "r") as file:
            data = json.load(file)
        return data
    except:
        print("settings failed to load")

def save_settings(info_tab_on_launch, backbone, optimizer, fp16):
    try:
        with open("settings.json", "w") as file:
            settings["info_tab_on_launch"] = info_tab_on_launch
            settings["backbone"] = backbone
            settings["optimizer"] = optimizer
            settings["fp16"] = fp16
            
            json.dump(settings, file, indent=4)
    except:
        print("settings failed to save")

def load_project(project_name):
    global max_epoch
    os.makedirs("temp/{}".format(project_name), exist_ok=True)

    items = os.listdir("workspaces/{}/results".format(project_name))
    if not items:
        return gr.Slider.update(maximum=0, value=0)
    for item in items:
        index = str(item).find("df_ep")+5
        if index > 0:
            max_epoch = int(str(item)[index:index+4])
            break
    
    video = cv2.VideoCapture("workspaces/{}/results/df_ep{:04d}_rgb.mp4".format(project_name, max_epoch))
    image = video.read()[1]
    cv2.imwrite("temp/{}/project_manager/image_{:02d}.png".format(project_name, 0), image)
    try:
        for i in range(1, 100):
            temp = video.read()[1]
            cv2.imwrite("temp/{}/project_manager/image_{:02d}.png".format(project_name, i), temp)
    except:
        print("video doesn't have 100 frames")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    with zipfile.ZipFile("temp/{}/{}.zip".format(project_name, project_name), "w") as file:
        file.write("workspaces/{}/mesh/albedo.png".format(project_name), arcname="albedo.png")
        file.write("workspaces/{}/mesh/mesh.mtl".format(project_name), arcname="mesh.mtl")
        file.write("workspaces/{}/mesh/mesh.obj".format(project_name), arcname="mesh.obj")
    
    return image, gr.Slider(label="slide to change viewpoint of model", minimum=0, maximum=99, value=0, step=1), \
           gr.Button(value="delete", visible=True, variant="stop"), gr.File(value="temp/{}/{}.zip".format(project_name, project_name), label="download", visible=True)

def delete_project(project_name):
    # delete_directory("workspaces/"+project_name)
    return gr.Image(value=None, interactive=False), gr.Slider(minimum=0, maximum=0, value=None, step=1, label="slide to change viewpoint of model"), \
           gr.Dropdown(choices=os.listdir("workspaces"), value=None, label="project name"), gr.Button(visible=False), gr.File(visible=False)
                        

# gradio event functions

def previous_tab_button_handler():
    global current_tab
    if current_tab > 3:
        current_tab -= 1
    return gr.Tabs(selected=current_tab)

def next_tab_button_handler():
    global current_tab
    if current_tab < 7:
        current_tab += 1
    return gr.Tabs(selected=current_tab)

def preprocess_image_button_handler(project_name_input, image_input):
    global project_name
    project_name = project_name_input
    image = preprocess(image_input, size=512)[0]
    os.makedirs("temp/{}".format(project_name), exist_ok=True)
    cv2.imwrite("temp/{}/image.png".format(project_name), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    return image

def return_input_image_handler():
    image = cv2.cvtColor(cv2.imread("temp/{}/image.png".format(project_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    return image

def images_viewer_slider_handler(slider):
    # updates the image based on the slider value, usually to select the "angle" (index of the image)
    global current_tab
    global project_name
    global radius
    if current_tab == 4:
        radius = slider
        image = cv2.imread("temp/{}/radius_discovery/image_{:.1f}.png".format(project_name, slider))
    elif current_tab == 5:
        image = cv2.imread("temp/{}/six_view_generation/image_{:1}.png".format(project_name, slider))
    else:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def images_viewer_slider_project_manager_handler(project_name, slider):
    # updates the image based on the slider value, usually to select the "angle" (index of the image)
    image = cv2.imread("temp/{}/project_manager/image_{:02d}.png".format(project_name, slider))
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

# zero123/stable-dreamfusion/cv functions

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
    precision_scope = torch.autocast if precision == 'autocast' else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            
            
            T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z])
            
            
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im)).mode().detach().repeat(n_samples, 1, 1, 1)]
            
            
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None
            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=cond, batch_size=n_samples, shape=shape, verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=ddim_eta, x_T=None)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def generate_novel_views(image, iters, polars, azimuths, radii, size=256):
    # polars, top = -90, straight = 0, bottom = 90
    # azimuth, left = -90, front = 0, right = 90, behind = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = zero123_utils.load_model_from_config(OmegaConf.load("pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"), "pretrained/zero123/zero123-xl.ckpt", device)
    model = zero123_utils.load_model_from_config(OmegaConf.load("pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"), "pretrained/zero123/105000.ckpt", device)
    model.use_ema = False
    
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    image = transforms.functional.resize(image, [size, size], antialias=True)
    
    sampler = DDIMSampler(model)
    images = []
    for i in range(len(polars)):
        x_samples_ddim = sample_model(image, model, sampler, "fp32", size, size, iters, 1, 3.0, 1.0, polars[i], azimuths[i], radii[i])
        novel_image = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            novel_image.append(np.asarray(x_sample.astype(np.uint8)))
            images.append(novel_image[0])
        del x_samples_ddim
        del novel_image
        clear_memory()
    del radii
    del model
    del image
    del device
    del sampler
    clear_memory()
    
    return images

def generate_model(opt):
    global max_epoch
    
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
    
    # save max_epoch
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    
    # starts the training
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)
    
    # saves mesh
    trainer.save_mesh()

def radius_discovery_handler():
    global project_name
    os.makedirs("temp/{}/radius_discovery".format(project_name), exist_ok=True)
    polars = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    azimuths = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    radii = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    image = cv2.cvtColor(cv2.imread("temp/{}/image.png".format(project_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB)
    images = generate_novel_views(image, 100, polars, azimuths, radii)

    for i in range(10):
        images[i] = cv2.resize(images[i], (512, 512))
        cv2.imwrite("temp/{}/radius_discovery/image_{:.1f}.png".format(project_name, radii[i]), cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
    
    return images[0], gr.Slider(minimum=0.0, maximum=3.6, label="slide to view images generated with different input radius", step=0.4, interactive=True)

def six_view_generation_handler():
    global project_name
    global radius
    os.makedirs("temp/{}/six_view_generation".format(project_name), exist_ok=True)
    polars = [0.0, 0.0, 0.0, 0.0, -90.0, 90.0]
    azimuths = [0.0, 0.0, 180.0, -90.0, 0.0, 0.0]
    radii = [radius, radius, radius, radius, radius, radius]
    image = cv2.cvtColor(cv2.imread("temp/{}/image.png".format(project_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB)
    images = generate_novel_views(image, 200, polars, azimuths, radii)
    
    for i in range(6):
        images[i] = cv2.resize(images[i], (512, 512))
        cv2.imwrite("temp/{}/six_view_generation/image_{:1}.png".format(project_name, i), cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
    
    return images[0], gr.Slider(minimum=0, maximum=5, label="slide to view images generated from different angles", step=1, interactive=True)

def model_reconstruction_handler(seed, size, iters, lr, batch_size, dataset_size_train, dataset_size_valid, dataset_size_test):
    global settings
    global current_tab
    global project_name
    global radius
    
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
        
    # these arguments are for zero123
    opt["backbone"] = settings["backbone"]
    opt["optim"] = settings["optimizer"]
    opt["fp16"] = settings["fp16"]
    opt["workspace"] = "workspaces/{}".format(project_name)
    opt["seed"] = seed
    opt["h"] = size
    opt["w"] = size
    opt["iters"] = iters
    opt["lr"] = lr
    opt["batch_size"] = batch_size
    opt["dataset_size_train"] = dataset_size_train
    opt["dataset_size_valid"] = dataset_size_valid
    opt["dataset_size_test"] = dataset_size_test
    opt["images"] = ["temp/{}/model_reconstruction/image_0_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_1_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_2_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_3_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_4_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_5_rgba.png".format(project_name)]
    opt["ref_polars"] = [90.0, 90.0, 90.0, 90.0, 180.0, 0.0001]
    opt["ref_azimuths"] = [0.0, 90.0, 180.0, -90.0, 0.0, 0.0]
    opt["ref_radii"] = [radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001]
    opt["zero123_ws"] = [16, 4, 1, 4, 2, 2]
    opt["exp_start_iter"] = opt["exp_start_iter"] or 0
    opt["exp_end_iter"] = opt["exp_end_iter"] or opt["iters"]
    opt = argparse.Namespace(**opt)
    
    # preprocess and save images to temporary folder in order to make calls to stable-dreamfusion trainer without making changes to its code
    os.makedirs("temp/{}/model_reconstruction".format(project_name), exist_ok=True)
    for i in range(6):
        image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_{:1}.png".format(project_name, i)), cv2.COLOR_BGRA2RGBA)
        image_rgba, image_depth, image_normal = preprocess(image, size=1024)
        cv2.imwrite("temp/{}/model_reconstruction/image_{:1}_rgba.png".format(project_name, i), cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite("temp/{}/model_reconstruction/image_{:1}_depth.png".format(project_name, i), image_depth)
        cv2.imwrite("temp/{}/model_reconstruction/image_{:1}_normal.png".format(project_name, i), image_normal)
        del image_rgba, image_depth, image_normal
        
    generate_model(opt)
    
    video = cv2.VideoCapture("workspaces/{}/results/df_ep{:04d}_rgb.mp4".format(project_name, max_epoch))
    image = video.read()[1]
    cv2.imwrite("workspaces/{}/results/df_ep{:04d}_rgb.mp4".format(project_name, max_epoch, 0), image)
    for i in range(1, dataset_size_test):
        temp = video.read()[1]
        cv2.imwrite("workspaces/{}/results/df_ep{:04d}_rgb.mp4".format(project_name, max_epoch, i), temp)
        
    with zipfile.ZipFile("temp/{}/{}.zip".format(project_name, project_name), "w") as file:
        file.write("workspaces/{}/mesh/albedo.png".format(project_name), arcname="albedo.png")
        file.write("workspaces/{}/mesh/mesh.mtl".format(project_name), arcname="mesh.mtl")
        file.write("workspaces/{}/mesh/mesh.obj".format(project_name), arcname="mesh.obj")
        
    data = {}
    with open("workspaces/{}/info.json".format(project_name), "w") as file:
        data["project_name"] = project_name
        data["seed"] = seed
        data["max_epoch"] = max_epoch
        data["backbone"] = settings["backbone"]
        data["optim"] = settings["optimizer"]
        data["fp16"] = settings["fp16"]
        data["size"] = size
        data["iters"] = iters
        data["lr"] = lr
        data["batch_size"] = batch_size
        data["dataset_size_train"] = dataset_size_train
        data["dataset_size_valid"] = dataset_size_valid
        data["dataset_size_test"] = dataset_size_test
        json.dump(data, file, indent=4)
        
    del data
    
    clear_memory()
    
    return image, gr.Slider(minimum=0, maximum=dataset_size_test, label="slide to view generated model from different angles", step=1, interactive=True), gr.File.update(value="temp/{}/{}.zip".format(project_name, project_name), visible=True)

def model_finetuning_handler(seed, size, tet_grid_size, iters, lr, batch_size, dataset_size_train, dataset_size_valid, dataset_size_test):
    global settings
    global current_tab
    global project_name
    global radius
    
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
        
    # these arguments are for zero123
    opt["backbone"] = settings["backbone"]
    opt["optim"] = settings["optimizer"]
    opt["fp16"] = settings["fp16"]
    opt["workspace"] = "workspaces/{}_dmtet".format(project_name)
    opt["seed"] = seed
    opt["h"] = size
    opt["w"] = size
    opt["iters"] = iters
    opt["lr"] = lr
    opt["batch_size"] = batch_size
    opt["dataset_size_train"] = dataset_size_train
    opt["dataset_size_valid"] = dataset_size_valid
    opt["dataset_size_test"] = dataset_size_test
    opt["images"] = ["temp/{}/model_reconstruction/image_0_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_1_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_2_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_3_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_4_rgba.png".format(project_name), "temp/{}/model_reconstruction/image_5_rgba.png".format(project_name)]
    opt["ref_polars"] = [90.0, 90.0, 90.0, 90.0, 180.0, 0.0001]
    opt["ref_azimuths"] = [0.0, 90.0, 180.0, -90.0, 0.0, 0.0]
    opt["ref_radii"] = [radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001, radius+0.0001]
    opt["zero123_ws"] = [16, 4, 1, 4, 2, 2]
    opt["exp_start_iter"] = opt["exp_start_iter"] or 0
    opt["exp_end_iter"] = opt["exp_end_iter"] or opt["iters"]
    opt["dmtet"] = True
    opt["init_with"] = "workspaces/{}/checkpoints/df.pth".format(project_name)
    opt["tet_grid_size"] = tet_grid_size
    opt.pop("full_radius_range")
    opt.pop("full_theta_range")
    opt.pop("full_phi_range")
    opt.pop("full_fovy_range")
    opt = argparse.Namespace(**opt)
    
    generate_model(opt)
    
    video = cv2.VideoCapture("workspaces/{}_dmtet/results/df_ep{:04d}_rgb.mp4".format(project_name, max_epoch))
    image = video.read()[1]
    cv2.imwrite("workspaces/{}_dmtet/results/df_ep{:04d}_rgb.mp4".format(project_name, max_epoch, 0), image)
    for i in range(1, dataset_size_test):
        temp = video.read()[1]
        cv2.imwrite("workspaces/{}_dmtet/results/df_ep{:04d}_rgb.mp4".format(project_name, max_epoch, i), temp)
        
    with zipfile.ZipFile("temp/{}_dmtet/{}_dmtet.zip".format(project_name, project_name), "w") as file:
        file.write("workspaces/{}_dmtet/mesh/albedo.png".format(project_name), arcname="albedo.png")
        file.write("workspaces/{}_dmtet/mesh/mesh.mtl".format(project_name), arcname="mesh.mtl")
        file.write("workspaces/{}_dmtet/mesh/mesh.obj".format(project_name), arcname="mesh.obj")
        
    data = {}
    with open("workspaces/{}_dmtet/info.json".format(project_name), "w") as file:
        data["project_name"] = "{}_dmtet".format(project_name)
        data["seed"] = seed
        data["max_epoch"] = max_epoch
        data["backbone"] = settings["backbone"]
        data["optim"] = settings["optimizer"]
        data["fp16"] = settings["fp16"]
        data["size"] = size
        data["iters"] = iters
        data["lr"] = lr
        data["batch_size"] = batch_size
        data["dataset_size_train"] = dataset_size_train
        data["dataset_size_valid"] = dataset_size_valid
        data["dataset_size_test"] = dataset_size_test
        json.dump(data, file, indent=4)
        
    del data
    
    clear_memory()
    
    return image, gr.Slider(minimum=0, maximum=dataset_size_test, label="slide to view generated model from different angles", step=1, interactive=True), gr.File.update(value="temp/{}/{}.zip".format(project_name, project_name), visible=True)

    
if __name__ == "__main__":
    main()