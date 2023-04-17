### 2023.4.7
Improvement on mesh quality & DMTet finetuning support.

https://user-images.githubusercontent.com/25863658/230535363-298c960e-bf9c-4906-8b96-cd60edcb24dd.mp4

### 2023.3.30
* adopt ideas from [Fantasia3D](https://fantasia3d.github.io/) to concatenate normal and mask as the latent code in a warm up stage, which shows faster convergence of shape.

https://user-images.githubusercontent.com/25863658/230535373-6ee28f16-bb21-4ec4-bc86-d46597361a04.mp4

### 2023.1.30
* Use an MLP to predict the surface normals as in Magic3D to avoid finite difference / second order gradient, generation quality is greatly improved.
* More efficient two-pass raymarching in training inspired by nerfacc.

https://user-images.githubusercontent.com/25863658/215996308-9fd959f5-b5c7-4a8e-a241-0fe63ec86a4a.mp4

### 2022.12.3
* Support Stable-diffusion 2.0 base.

### 2022.11.15
* Add the vanilla backbone that is pure-pytorch.

### 2022.10.9
* The shading (partially) starts to work, at least it won't make scene empty. For some prompts, it shows better results (less severe Janus problem). The textureless rendering mode is still disabled.
* Enable shading by default (--warmup_iters 1000).

### 2022.10.5
* Basic reproduction finished.
* Non --cuda_ray, --tcnn are not working, need to fix.
* Shading is not working, disabled in utils.py for now. Surface normals are bad.
* Use an entropy loss to regularize weights_sum (alpha), the original L2 reg always leads to degenerated geometry...

https://user-images.githubusercontent.com/25863658/194241493-f3e68f78-aefe-479e-a4a8-001424a61b37.mp4
