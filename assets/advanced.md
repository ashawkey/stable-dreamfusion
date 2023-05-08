
# Code organization & Advanced tips

This is a simple description of the most important implementation details.
If you are interested in improving this repo, this might be a starting point.
Any contribution would be greatly appreciated!

* The SDS loss is located at `./guidance/sd_utils.py > StableDiffusion > train_step`:
```python
## 1. we need to interpolate the NeRF rendering to 512x512, to feed it to SD's VAE.
pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
## 2. image (512x512) --- VAE --> latents (64x64), this is SD's difference from Imagen.
latents = self.encode_imgs(pred_rgb_512)
... # timestep sampling, noise adding and UNet noise predicting
## 3. the SDS loss
w = (1 - self.alphas[t])
grad = w * (noise_pred - noise)
# since UNet part is ignored and cannot simply audodiff, we have two ways to set the grad:
# 3.1. call backward and set the grad now (need to retain graph since we will call a second backward for the other losses later)
latents.backward(gradient=grad, retain_graph=True)
return 0 # dummy loss
# 3.2. use a custom function to set a hook in backward, so we only call backward once (credits to @elliottzheng)
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

loss = SpecifyGradient.apply(latents, grad)
return loss # functional loss
```
* Other regularizations are in `./nerf/utils.py > Trainer > train_step`.
    * The generation seems quite sensitive to regularizations on weights_sum (alphas for each ray). The original opacity loss tends to make NeRF disappear (zero density everywhere), so we use an entropy loss to replace it for now (encourages alpha to be either 0 or 1).
* NeRF Rendering core function: `./nerf/renderer.py > NeRFRenderer > run & run_cuda`.
* Shading & normal evaluation: `./nerf/network*.py > NeRFNetwork > forward`.
    * light direction: current implementation use a plane light source, instead of a point light source.
* View-dependent prompting: `./nerf/provider.py > get_view_direction`.
    * use `--angle_overhead, --angle_front` to set the border.
* Network backbone (`./nerf/network*.py`) can be chosen by the `--backbone` option.
* Spatial density bias (density blob): `./nerf/network*.py > NeRFNetwork > density_blob`.


# Debugging

`debugpy-run` is a convenient way to remotely debug this project. Simply replace a command like this one:

```bash
python main.py --text "a hamburger" --workspace trial -O --vram_O
```

... with:

```bash
debugpy-run main.py -- --text "a hamburger" --workspace trial -O --vram_O
```

For more details: https://github.com/bulletmark/debugpy-run 
