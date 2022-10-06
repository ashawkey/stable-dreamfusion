### 2022.10.5
* Basic reproduction finished.
* Non --cuda_ray, --tcnn are not working, need to fix.
* Shading is not working, disabled in utils.py for now. Surface normals are bad.
* Use an entropy loss to regularize weights_sum (alpha), the original L2 reg always leads to degenerated geometry...