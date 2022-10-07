# Stable-Dreamfusion

A pytorch implementation of the text-to-3D model **Dreamfusion**, powered by the [Stable Diffusion](https://github.com/CompVis/stable-diffusion) text-to-2D model.

The original paper's project page: [_DreamFusion: Text-to-3D using 2D Diffusion_](https://dreamfusion3d.github.io/).

Colab notebook for usage: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MXT3yfOFvO0ooKEfiUUvTKwUkrrlCHpF?usp=sharing)

Examples generated from text prompt `a high quality photo of a pineapple` viewed with the GUI in real time:

https://user-images.githubusercontent.com/25863658/194241493-f3e68f78-aefe-479e-a4a8-001424a61b37.mp4

### [Gallery](https://github.com/ashawkey/stable-dreamfusion/issues/1) | [Update Logs](assets/update_logs.md)

# Important Notice
This project is a **work-in-progress**, and contains lots of differences from the paper. Also, many features are still not implemented now. **The current generation quality cannot match the results from the original paper, and many prompts still fail badly!** 


## Notable differences from the paper
* Since the Imagen model is not publicly available, we use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) to replace it (implementation from [diffusers](https://github.com/huggingface/diffusers)). Different from Imagen, Stable-Diffusion is a latent diffusion model, which diffuses in a latent space instead of the original image space. Therefore, we need the loss to propagate back from the VAE's encoder part too, which introduces extra time cost in training. Currently, 15000 training steps take about 5 hours to train on a V100.
* We use the [multi-resolution grid encoder](https://github.com/NVlabs/instant-ngp/) to implement the NeRF backbone (implementation from [torch-ngp](https://github.com/ashawkey/torch-ngp)), which enables much faster rendering (~10FPS at 800x800).
* We use the Adam optimizer with a larger initial learning rate.


## TODOs
* The normal evaluation & shading part.
* Better mesh (improve the surface quality). 

# Install

```bash
git clone https://github.com/ashawkey/stable-dreamfusion.git
cd stable-dreamfusion
```

**Important**: To download the Stable Diffusion model checkpoint, you should create a file called `TOKEN` under this directory (i.e., `stable-dreamfusion/TOKEN`) and copy your hugging face [access token](https://huggingface.co/docs/hub/security-tokens) into it.

### Install with pip
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone if using --tcnn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# (optional) install CLIP guidance for the dreamfield setting
pip install git+https://github.com/openai/CLIP.git

# (optional) install nvdiffrast for exporting textured mesh
pip install git+https://github.com/NVlabs/nvdiffrast/
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
### stable-dreamfusion setting
## train with text prompt
# `-O` equals `--cuda_ray --fp16 --dir_text`
python main.py --text "a hamburger" --workspace trial -O

## after the training is finished:
# test (exporting 360 video, and an obj mesh with png texture)
python main.py --workspace trial -O --test

# test with a GUI (free view control!)
python main.py --workspace trial -O --test --gui

### dreamfields (CLIP) setting
python main.py --text "a hamburger" --workspace trial_clip -O --guidance clip
python main.py --text "a hamburger" --workspace trial_clip -O --test --gui --guidance clip
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
w = (1 - self.scheduler.alphas_cumprod[t]).to(self.device)
grad = w * (noise_pred - noise)
latents.backward(gradient=grad, retain_graph=True)
```
* Other regularizations are in `./nerf/utils.py > Trainer > train_step`. 
    * The generation seems quite sensitive to regularizations on weights_sum (alphas for each ray). The original opacity loss tends to make NeRF disappear (zero density everywhere), so we use an entropy loss to replace it for now (encourages alpha to be either 0 or 1).
* NeRF Rendering core function: `./nerf/renderer.py > NeRFRenderer > run_cuda`.
    * the occupancy grid based training acceleration (instant-ngp like, enabled by `--cuda_ray`) may harm the generation progress, since once a grid cell is marked as empty, rays won't pass it later...
* Shading & normal evaluation: `./nerf/network*.py > NeRFNetwork > forward`. Current implementation harms training and is disabled.
    * use `--albedo_iters 1000` to enable random shading mode after 1000 steps from albedo, lambertian, and textureless.
    * light direction: current implementation use a plane light source, instead of a point light source...
* View-dependent prompting: `./nerf/provider.py > get_view_direction`.
    * ues `--angle_overhead, --angle_front` to set the border. How to better divide front/back/side regions?
* Network backbone (`./nerf/network*.py`) can be chosen by the `--backbone` option, but `tcnn` and `vanilla` are not well tested.
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
