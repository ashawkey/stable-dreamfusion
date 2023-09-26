# stable-dreamfusion-gui

A gradio interface to use [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion).

Currently focusing on [Zero123](https://github.com/cvlab-columbia/zero123), will add support to other components later.

# Install
This installation guide assumes you are running this on a Windows machine.

Download and install [Git for Windows](https://git-scm.com/download/win) and run these in the command prompt.

```bash
git clone https://github.com/ghotinggoad/stable-dreamfusion-gui.git
cd stable-dreamfusion-gui
```

### Installing CUDA and MSVC

Download and install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-toolkit-archive) and [MSVC v143](https://visualstudio.microsoft.com/downloads/).

For MSVC v143, download Visual Studio 2022 Community Edition and make sure MSVC v143 is checked during installation.

Afterwards, CUDA drivers will be built during pip installations.

### Optional: create a python virtual environment

To avoid python package conflicts, we recommend using a virtual environment, e.g.: using conda or venv:

```bash
cd stable-dreamfusion-gui
python -m venv venv_stable-dreamfusion-gui
source venv_stable-dreamfusion-gui\bin\activate # you need to repeat this step for every new terminal
```

### Install with pip

```bash
cd stable-dreamfusion-gui
pip install -r requirements.txt
```

### Download pre-trained models

To use image-conditioned 3D generation, you need to download some pretrained checkpoints manually:
* [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) for diffusion backend.
    
    We use [zero123-xl.ckpt](https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt) by default, and it is hard-coded in "guidance/zero123_utils.py".
    
    Download the ckpt file and put it in "pretrained/zero123"
* [Omnidata](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch) for [depth](https://drive.google.com/uc?id=1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t) and [normal](https://drive.google.com/uc?id=1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t) prediction.
    
    These ckpts are hardcoded in "preprocess_image.py".
    
    Download the ckpt files and put them in "pretrained/omnidata".

For DMTet, we port the pre-generated "32/64/128" resolution tetrahedron grids under "tets".
The 256 resolution one can be found [here](https://drive.google.com/file/d/1lgvEKNdsbW5RS4gVxJbgBS4Ac92moGSa/view?usp=sharing).

### Build extension (optional)
By default, we use ["load"](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
We also provide the "setup.py" to build each extension:
```bash
cd stable-dreamfusion-gui

# install all extension modules
.\scripts\install_ext.bat

# if you want to install manually, here is an example:
pip install .\raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```

### Tested environments
* Windows 11 Pro with torch 2.0.1 & CUDA 11.8 on an RTX3090.

# Usage
```bash
cd stable-dreamfusion-gui
python gradio_app.py
```

Open "127.0.0.1:7860" in your browser and start exploring!
* First time running will take some time to compile the CUDA extensions.
* Console might print an error stating that port 7860 is already in use, just ignore the warning and open the given IP Address on your browser.

Press "ctrl+c" to quit, make sure to only press once so that gradio can exit properly. Otherwise, the port might not be released and you get the error stated above.

# Acknowledgement

This work is largely based on ashawkey's [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) for which I take no credit for, thank you to ashawkey!