### 2022.12.3
* Support Stable-diffusion 2.0 base.

### 2022.11.15
* Add the vanilla backbone that is pure-pytorch.

### 2022.10.9
* The shading (partially) starts to work, at least it won't make scene empty. For some prompts, it shows better results (less severe Janus problem). The textureless rendering mode is still disabled.
* Enable shading by default (--albedo_iters 1000).

### 2022.10.5
* Basic reproduction finished.
* Non --cuda_ray, --tcnn are not working, need to fix.
* Shading is not working, disabled in utils.py for now. Surface normals are bad.
* Use an entropy loss to regularize weights_sum (alpha), the original L2 reg always leads to degenerated geometry...