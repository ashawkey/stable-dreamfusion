# Stable-Dreamfusion

A pytorch implementation of the text-to-3D model **Dreamfusion**, powered by the [Stable Diffusion](https://github.com/CompVis/stable-diffusion) text-to-2D model.

The original paper's project page: [_DreamFusion: Text-to-3D using 2D Diffusion_](https://dreamfusion3d.github.io/).

**NEW**: Stable-diffusion 2.0 base is supported!

Colab notebooks: 
* Instant-NGP backbone (`-O`): [![Instant-NGP Backbone](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MXT3yfOFvO0ooKEfiUUvTKwUkrrlCHpF?usp=sharing)

* Vanilla NeRF backbone (`-O2`): [![Vanilla Backbone](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mvfxG-S_n_gZafWoattku7rLJ2kPoImL?usp=sharing) 

Examples generated from text prompt `a high quality photo of a pineapple` viewed with the GUI in real time:

https://user-images.githubusercontent.com/25863658/194241493-f3e68f78-aefe-479e-a4a8-001424a61b37.mp4

### [Gallery](https://github.com/ashawkey/stable-dreamfusion/issues/1) | [Update Logs](assets/update_logs.md)

# Important Notice
This project is a **work-in-progress**, and contains lots of differences from the paper. Also, many features are still not implemented now. **The current generation quality cannot match the results from the original paper, and many prompts still fail badly!** 

## Notable differences from the paper
* Since the Imagen model is not publicly available, we use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) to replace it (implementation from [diffusers](https://github.com/huggingface/diffusers)). Different from Imagen, Stable-Diffusion is a latent diffusion model, which diffuses in a latent space instead of the original image space. Therefore, we need the loss to propagate back from the VAE's encoder part too, which introduces extra time cost in training. Currently, 10000 training steps take about 3 hours to train on a V100.
* We use the [multi-resolution grid encoder](https://github.com/NVlabs/instant-ngp/) to implement the NeRF backbone (implementation from [torch-ngp](https://github.com/ashawkey/torch-ngp)), which enables much faster rendering (~10FPS at 800x800). The vanilla NeRF backbone is also supported now, but the Mip-NeRF backbone as the paper is still not implemented.
* We use the Adam optimizer.


## The multi-face [Janus problem](https://twitter.com/poolio/status/1578045212236034048).
* This is likely to be caused by the text-to-2D model's capability, as discussed by [Magic3D](https://deepimagination.cc/Magic3D/) in Figure 4 and *Can single-stage optimization work with LDM prior?*.


# Install

```bash
git clone https://github.com/ashawkey/stable-dreamfusion.git
cd stable-dreamfusion
```

### Install with pip
```bash
pip install -r requirements.txt

# (optional) install nvdiffrast for exporting textured mesh (if use --save_mesh)
pip install git+https://github.com/NVlabs/nvdiffrast/

# (optional) install CLIP guidance for the dreamfield setting
pip install git+https://github.com/openai/CLIP.git

```

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
We also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```

### Tested environments
* Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.


# Usage

First time running will take some time to compile the CUDA extensions.

```bash
#### stable-dreamfusion setting

### Instant-NGP NeRF Backbone 
# + faster rendering speed
# + less GPU memory (~16G)
# - need to build CUDA extensions
# - worse surface quality

## train with text prompt (with the default settings)
# `-O` equals `--cuda_ray --fp16 --dir_text`
# `--cuda_ray` enables instant-ngp-like occupancy grid based acceleration.
# `--fp16` enables half-precision training.
# `--dir_text` enables view-dependent prompting.
python main.py --text "a hamburger" --workspace trial -O

# choose stable-diffusion version (support 1.5 and 2.0, default is 2.0 now)
python main.py --text "a hamburger" --workspace trial -O --sd_version 1.5

# we also support negative text prompt now:
python main.py --text "a rose" --negative "red" --workspace trial -O

## if the above command fails to generate meaningful things (learns an empty scene), maybe try:
# 1. disable random lambertian/textureless shading, simply use albedo as color:
python main.py --text "a hamburger" --workspace trial -O --albedo
# 2. use a smaller density regularization weight:
python main.py --text "a hamburger" --workspace trial -O --lambda_entropy 1e-5

# you can also train in a GUI to visualize the training progress:
python main.py --text "a hamburger" --workspace trial -O --gui

# A Gradio GUI is also possible (with less options):
python gradio_app.py # open in web browser

## after the training is finished:
# test (exporting 360 degree video)
python main.py --workspace trial -O --test
# also save a mesh (with obj, mtl, and png texture)
python main.py --workspace trial -O --test --save_mesh
# test with a GUI (free view control!)
python main.py --workspace trial -O --test --gui

### Vanilla NeRF backbone
# + better surface quality
# + pure pytorch, no need to build extensions!
# - slow rendering speed
# - more GPU memory

## train
# `-O2` equals `--dir_text --backbone vanilla`
python main.py --text "a hotdog" --workspace trial2 -O2

## if CUDA OOM, maybe try:
# 1. only use albedo rendering, less GPU memory (~16G), train faster, but results may be worse
python main.py --text "a hotdog" --workspace trial2 -O2 --albedo
# 2. reduce NeRF sampling steps (--num_steps and --upsample_steps)
python main.py --text "a hotdog" --workspace trial2 -O2 --num_steps 64 --upsample_steps 0

## test
python main.py --workspace trial2 -O2 --test
python main.py --workspace trial2 -O2 --test --save_mesh
python main.py --workspace trial2 -O2 --test --gui # not recommended, FPS will be low.
```

# Code organization & Advanced tips

This is a simple description of the most important implementation details. 
If you are interested in improving this repo, this might be a starting point.
Any contribution would be greatly appreciated!

* The SDS loss is located at `./nerf/sd.py > StableDiffusion > train_step`:
```python
# 1. we need to interpolate the NeRF rendering to 512x512, to feed it to SD's VAE.
pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
# 2. image (512x512) --- VAE --> latents (64x64), this is SD's difference from Imagen.
latents = self.encode_imgs(pred_rgb_512)
... # timestep sampling, noise adding and UNet noise predicting
# 3. the SDS loss, since UNet part is ignored and cannot simply audodiff, we manually set the grad for latents.
w = (1 - self.alphas[t])
grad = w * (noise_pred - noise)
latents.backward(gradient=grad, retain_graph=True)
```
* Other regularizations are in `./nerf/utils.py > Trainer > train_step`. 
    * The generation seems quite sensitive to regularizations on weights_sum (alphas for each ray). The original opacity loss tends to make NeRF disappear (zero density everywhere), so we use an entropy loss to replace it for now (encourages alpha to be either 0 or 1).
* NeRF Rendering core function: `./nerf/renderer.py > NeRFRenderer > run & run_cuda`.
* Shading & normal evaluation: `./nerf/network*.py > NeRFNetwork > forward`. Current implementation harms training and is disabled.
    * light direction: current implementation use a plane light source, instead of a point light source.
* View-dependent prompting: `./nerf/provider.py > get_view_direction`.
    * use `--angle_overhead, --angle_front` to set the border.
    * use `--suppress_face` to add `face` as a negative prompt at all directions except `front`.
* Network backbone (`./nerf/network*.py`) can be chosen by the `--backbone` option.
* Spatial density bias (gaussian density blob): `./nerf/network*.py > NeRFNetwork > gaussian`.

# Acknowledgement

* The amazing original work: [_DreamFusion: Text-to-3D using 2D Diffusion_](https://dreamfusion3d.github.io/).
    ```
    @article{poole2022dreamfusion,
        author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
        title = {DreamFusion: Text-to-3D using 2D Diffusion},
        journal = {arXiv},
        year = {2022},
    }
    ```

* Huge thanks to the [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and the [diffusers](https://github.com/huggingface/diffusers) library. 

    ```
    @misc{rombach2021highresolution,
        title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
        author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
        year={2021},
        eprint={2112.10752},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

    @misc{von-platen-etal-2022-diffusers,
        author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
        title = {Diffusers: State-of-the-art diffusion models},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/huggingface/diffusers}}
    }
    ```

* The GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).

# Citation

If you find this work useful, a citation will be appreciated via:
```
@misc{stable-dreamfusion,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/stable-dreamfusion},
    Title = {Stable-dreamfusion: Text-to-3D with Stable-diffusion}
}
```