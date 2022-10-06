# Stable-Dreamfusion

A pytorch implementation of the text-to-3D model **Dreamfusion**, powered by the [Stable Diffusion](https://github.com/CompVis/stable-diffusion) text-to-2D model.

The original paper's project page: [_DreamFusion: Text-to-3D using 2D Diffusion_](https://dreamfusion3d.github.io/).

Example of "a squierrel" and "a hamburger":

### [Gallery](assets/gallery.md) | [Update Logs](assets/update_logs.md)

# Important Notice
This project is a **work-in-progress**, and contains lots of differences from the paper. Also, many features are still not implmented now. The current generation quality cannot match the results from the original paper, and still fail badly for many prompts.

## Notable differences from the paper
* Since the Imagen model is not publicly available, we use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) to replace it (implementation from [diffusers](https://github.com/huggingface/diffusers)). Different from Imagen, Stable-Diffusion is a latent diffusion model, which diffuses in a latent space instead of the original image space. Therefore, we need the loss to propagate back from the VAE's encoder part too, which introduces extra time cost in training. Currently, 15000 training steps take about 5 hours to train on a V100.
* We use the [multi-resolution grid encoder](https://github.com/NVlabs/instant-ngp/) to implement the NeRF backbone (implementation from [torch-ngp](https://github.com/ashawkey/torch-ngp)), which enables much faster rendering (~10FPS at 800x800).
* We use the Adam optimizer with a larger initial learning rate.


## TODOs
* The shading part & normal evaluation.
* Exporting colored mesh.


# Install

```bash
git clone https://github.com/ashawkey/stable-dreamfusion.git
cd stable-dreamfusion
```

**Important**: To download the Stable Diffusion model checkpoint, you should create a file under this directory called `TOKEN` and copy your hugging face [access token](https://huggingface.co/docs/hub/security-tokens) into it.

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
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

### Tested environments
* Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.


# Usage

First time running will take some time to compile the CUDA extensions.

```bash
### stable-dreamfusion setting
# train with text prompt
# `-O` equals `--cuda_ray --fp16 --dir_text`
python main_nerf.py --text "a hamburger" --workspace trial -O

# test (exporting 360 video)
python main_nerf.py --text "a hamburger" --workspace trial -O --test

# test with a GUI (free view control!)
python main_nerf.py --text "a hamburger" --workspace trial -O --test --gui

### dreamfields (CLIP) setting
python main_nerf.py --text "a hamburger" --workspace trial_clip -O --guidance clip
python main_nerf.py --text "a hamburger" --workspace trial_clip -O --test --gui --guidance clip
```

# Acknowledgement

* The amazing original work: [_DreamFusion: Text-to-3D using 2D Diffusion_](https://dreamfusion3d.github.io/).

* Huge thanks to the [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and the [diffusers](https://github.com/huggingface/diffusers) library. 


* The GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).