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

# main

def main():
    # global variables used to parse variables between functions that are not users' direct input
    global settings
    global current_tab
    global max_epoch
    global project_name
    global image
    global radius

    with gr.Blocks(title="zero123d reconstruction") as app:
        # make temp directory
        os.makedirs("temp", exist_ok=True)
        
        # initialize global variables and configure gui based on loaded settings
        settings = load_settings()
        current_tab = 3
        if settings["info_tab_on_launch"]:
            current_tab = 0
        max_epoch = 0
        project_name = "temp"
        image = None
        radius = 0.0

        with gr.Tabs(selected=current_tab) as tabs:
            with gr.Tab(label="new project", id=3) as new_project_tab:
                # components
                project_name_input = gr.Textbox(label="project name (no special characters including spaces, only underscores)")
                with gr.Row():
                    image_input = gr.Image(shape=(512, 512), height=512, width=512, label="image")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="secondary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                new_project_tab.select(fn=lambda: globals().update(current_tab=3))
                project_name_input.input(fn=update_project_inputs_handler, inputs=[project_name_input, image_input])
                image_input.upload(fn=update_project_inputs_handler, inputs=[project_name_input, image_input]).then(fn=preprocess_input_image_handler, outputs=image_input)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
                
            with gr.Tab(label="radius discovery", id=4) as radius_discovery_tab:
                # components
                with gr.Row():
                    image_output = gr.Image(value=image, shape=(512, 512), height=512, width=512, interactive=False)
                    images_viewer_output = gr.Image(label="image")
                images_viewer_slider_input = gr.Slider(minimum=0.0, maximum=3.6, label="slide to view images generated with different input radius", step=0.4)
                generate_button = gr.Button(value="generate", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                radius_discovery_tab.select(fn=lambda: globals().update(current_tab=4)).then(fn=return_input_image_handler, outputs=image_output)
                generate_button.click(fn=radius_discovery_handler, inputs=image_input, outputs=images_viewer_output)
                images_viewer_slider_input.change(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
                
            with gr.Tab(label="six-view generation", id=5) as six_view_generation_tab:
                # components
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                six_view_generation_tab.select(fn=lambda: globals().update(current_tab=5)).then(fn=return_input_image_handler, outputs=image_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
            
            with gr.Tab(label="model reconstruction", id=6) as model_reconstruction_tab:
                # components
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                model_reconstruction_tab.select(fn=lambda: globals().update(current_tab=6)).then(fn=return_input_image_handler, outputs=image_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
            
            with gr.Tab(label="model fine-tuning", id=7) as model_fine_tuning_tab:
                # components
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="secondary")
                # events
                model_fine_tuning_tab.select(fn=lambda: globals().update(current_tab=7))
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
                model_viewer_slider_input.change(fn=model_viewer_slider_handler, inputs=[project_name_input, model_viewer_slider_input], outputs=model_viewer_output)
                project_name_input.input(fn=load_project, inputs=project_name_input, outputs=[model_viewer_output, model_viewer_slider_input, delete_button, file_output])
                delete_button.click(fn=delete_project, inputs=project_name_input, outputs=[model_viewer_output, model_viewer_slider_input, project_name_input, delete_button, file_output])
                
            with gr.Tab(label="settings", id=1) as settings_tab:
                # components
                info_tab_on_launch = gr.Checkbox(value=settings["info_tab_on_launch"], label="load up info tab on launch")
                settings_tab_save_button = gr.Button(value="save settings", variant="primary")
                # events
                settings_tab.select(fn=lambda: globals().update(current_tab=1))
                settings_tab_save_button.click(fn=save_settings, inputs=info_tab_on_launch)
                    
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
    
    app.launch()
    
    # rmdir temp folder
    delete_directory("temp")
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

def save_settings(info_tab_on_launch):
    try:
        with open("settings.json", "w") as file:
            settings["info_tab_on_launch"] = info_tab_on_launch
            json.dump(settings, file, indent=4)
    except:
        print("settings failed to save")

def load_project(project_name):
    global max_epoch
    os.makedirs("temp/"+project_name, exist_ok=True)

    items = os.listdir("workspaces/"+project_name+"/results")
    if not items:
        return gr.Slider.update(maximum=0, value=0)
    for item in items:
        index = str(item).find("df_ep")+5
        if index > 0:
            max_epoch = int(str(item)[index:index+4])
            break
    
    video = cv2.VideoCapture("workspaces/"+project_name+"/results/df_ep{:04d}_rgb.mp4".format(max_epoch))
    image = video.read()[1]
    
    cv2.imwrite("temp/"+project_name+"/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, 0), image)
    for i in range(1, 100):
        temp = video.read()[1]
        cv2.imwrite("temp/"+project_name+"/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, i), temp)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    with zipfile.ZipFile("temp/"+project_name+"/"+project_name+".zip", "w") as file:
        file.write("workspaces/"+project_name+"/mesh/albedo.png", arcname="albedo.png")
        file.write("workspaces/"+project_name+"/mesh/mesh.mtl", arcname="mesh.mtl")
        file.write("workspaces/"+project_name+"/mesh/mesh.obj", arcname="mesh.obj")
    
    return image, gr.Slider(label="slide to change viewpoint of model", minimum=0, maximum=99, value=0, step=1), \
           gr.Button(value="delete", visible=True, variant="stop"), gr.File(value="temp/"+project_name+"/"+project_name+".zip", label="download", visible=True)

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

def update_project_inputs_handler(project_name_input, image_input):
    global project_name
    global image
    project_name = project_name_input
    image = image_input

def preprocess_input_image_handler():
    global image
    image = preprocess(image)[0]
    return image

def return_input_image_handler():
    global image
    return gr.Image(value=image, label="image", interactive=False)

def images_viewer_slider_handler(slider):
    global project_name
    global radius
    radius = slider
    # updates the image based on the slider value, usually to select the "angle" (index of the image)
    image = cv2.imread("temp/"+project_name+"/radius_discovery/image_{:.1f}.png".format(slider))
    return gr.Image(value=image, interactive=False)

def model_viewer_slider_handler(project_name, slider):
    # updates the image based on the slider value, usually to select the "angle" (index of the image)
    image = cv2.imread("temp/"+project_name+"/df_ep{:04d}_{:02d}_rgb.png".format(max_epoch, slider))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return gr.Image(value=image, interactive=False)

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
            samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=cond, batch_size=n_samples, shape=shape, verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=ddim_eta, x_T=None)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def generate_novel_views(image, iters, polars, azimuths, radii):
    # polars, top = -90, straight = 0, bottom = 90
    # azimuth, left = -90, front = 0, right = 90, behind = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = zero123_utils.load_model_from_config(OmegaConf.load("pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"), "pretrained/zero123/zero123-xl.ckpt", device)
    model.use_ema = False
    
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    image = transforms.functional.resize(image, [256, 256], antialias=True)
    
    sampler = DDIMSampler(model)
    images = []
    for i in range(len(polars)):
        x_samples_ddim = sample_model(image, model, sampler, "fp16", 256, 256, iters, 1, 3.0, 1.0, polars[i], azimuths[i], radii[i])
        novel_image = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            novel_image.append(np.asarray(x_sample.astype(np.uint8)))
        images.append(novel_image[0])
        del x_samples_ddim
        del novel_image
        clear_memory()
    del radii
    del image
    del device
    del model
    del sampler
    clear_memory()
    
    return images

def radius_discovery_handler(image):
    global project_name
    os.makedirs("temp/"+project_name, exist_ok=True)
    os.makedirs("temp/"+project_name+"/radius_discovery", exist_ok=True)
    polars = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    azimuths = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    radii = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6]
    images = generate_novel_views(image, 100, polars, azimuths, radii)
    
    for i in range(10):
        cv2.imwrite("temp/"+project_name+"/radius_discovery/image_{:.1f}.png".format(radii[i]), images[i])
    
    return images[0]

def six_view_generation_handler(image):
    os.makedirs("temp/")

if __name__ == "__main__":
    main()