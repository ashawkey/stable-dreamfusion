import torch
import argparse
import pandas as pd
import sys

from nerf.provider import NeRFDataset
from nerf.utils import *

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='guidance model')
    parser.add_argument('--seed', default=None)

    parser.add_argument('--image', default=None, help="image prompt")
    parser.add_argument('--image_config', default=None, help="image config csv")

    parser.add_argument('--zero123_final', action='store_true', help="Use Zero123's final renders to train NeRF as ref view")

    parser.add_argument('--known_view_interval', type=int, default=2, help="train default view with RGB loss every & iters, only valid if --image is not None.")
    parser.add_argument('--guidance_scale', type=float, default=100, help="diffusion model classifier-free guidance scale")

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=256, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=5e4, help="target face number for mesh decimation")

    parser.add_argument('--dmtet', action='store_true', help="use dmtet finetuning")
    parser.add_argument('--tet_grid_size', type=int, default=128, help="tet grid size")
    parser.add_argument('--init_ckpt', type=str, default='', help="ckpt to init dmtet")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--taichi_ray', action='store_true', help="use taichi raymarching")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--warmup_iters', type=int, default=2000, help="training iters that only use albedo shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--grad_clip', type=float, default=-1, help="clip grad of all grad to this limit, negative value disables it")
    parser.add_argument('--grad_clip_rgb', type=float, default=-1, help="clip grad of rgb space grad to this limit, negative value disables it")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='softplus', choices=['softplus', 'exp'], help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=0.1, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=10, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.5, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--backbone', type=str, default='grid', choices=['grid_tcnn', 'grid', 'vanilla', 'grid_taichi'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--known_view_scale', type=float, default=1.5, help="multiply --h/w by this for known view rendering")
    parser.add_argument('--known_view_noise_scale', type=float, default=2e-3, help="random camera noise added to rays_o and rays_d")
    parser.add_argument('--dmtet_reso_scale', type=float, default=8, help="multiply --h/w by this for dmtet finetuning")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")

    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera fovy range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 80], help="training camera fovy range")

    parser.add_argument('--default_radius', type=float, default=1.2, help="radius for the default view")
    parser.add_argument('--default_theta', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_phi', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=60, help="fovy for the default view")
    parser.add_argument('--default_img_w', type=float, default=1, help="zero123 weight for the default view")

    parser.add_argument('--progressive_view', action='store_true', help="progressively expand view sampling range from default to full")
    parser.add_argument('--progressive_level', action='store_true', help="progressively increase gridencoder's max_level")

    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")

    parser.add_argument('--test_freq', type=int, default=None, help="Test model every n iterations")

    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")

    parser.add_argument('--lambda_mesh_normal', type=float, default=0.5, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_mesh_laplacian', type=float, default=0.5, help="loss scale for mesh laplacian")

    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--lambda_rgb', type=float, default=10, help="loss scale for RGB")
    parser.add_argument('--lambda_mask', type=float, default=5, help="loss scale for mask (alpha)")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for normal map")
    parser.add_argument('--lambda_depth', type=float, default=0.1, help="loss scale for relative depth")
    parser.add_argument('--lambda_2d_normal_smooth', type=float, default=0, help="loss scale for 2D normal image smoothness")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    parser.add_argument('--zero123_config', type=str, default='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml', help="config file for zero123")
    parser.add_argument('--zero123_ckpt', type=str, default='./pretrained/zero123/105000.ckpt', help="ckptfor zero123")

    parser.add_argument('--dataset_size_train', type=int, default=100, help="Length of train dataset")
    parser.add_argument('--dataset_size_valid', type=int, default=8, help="Length of train dataset")
    parser.add_argument('--dataset_size_test', type=int, default=100, help="Length of train dataset")

    parser.add_argument('--debug', action='store_true', help="debug options")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True

    elif opt.O2:
        opt.fp16 = True
        opt.backbone = 'vanilla'

    if opt.debug:
        opt.dataset_size_train = 1
        opt.dataset_size_valid = 1
        opt.dataset_size_test = 1
        opt.iters = 4
        opt.test_freq = 1

    opt.images, opt.radii, opt.thetas, opt.phis, opt.img_ws = [], [], [], [], []

    # parameters for image-conditioned generation
    if opt.image is not None or opt.image_config is not None:

        if opt.text is None:
            # use zero123 guidance model when only providing image
            opt.guidance = 'zero123'
            opt.fovy_range = [opt.default_fovy, opt.default_fovy] # fix fov as zero123 doesn't support changing fov

            # very important to keep the image's content
            opt.guidance_scale = 3
            opt.lambda_guidance = 0.02

        else:
            # use stable-diffusion when providing both text and image
            opt.guidance = 'stable-diffusion'

        opt.t_range = [0.02, 0.50]
        opt.lambda_orient = 10

        # latent warmup is not needed, we hardcode a 100-iter rgbd loss only warmup.
        opt.warmup_iters = 0

        # make shape init more stable
        opt.progressive_view = True
        opt.progressive_level = True

        if opt.image is not None:
            opt.images += [opt.image]
            opt.radii += [opt.default_radius]
            opt.thetas += [opt.default_theta]
            opt.phis += [opt.default_phi]
            opt.img_ws += [opt.default_img_w]

        if opt.image_config is not None:
            # for multiview (zero123)
            conf = pd.read_csv(opt.image_config, skipinitialspace=True)
            opt.images += list(conf.image)
            opt.radii += list(conf.radius)
            opt.thetas += list(conf.theta)
            opt.phis += list(conf.phi)
            opt.img_ws += list(conf.zero123_weight)
            if opt.image is None:
                opt.default_radius = opt.radii[0]
                opt.default_theta = opt.thetas[0]
                opt.default_phi = opt.phis[0]
                opt.default_img_w = opt.img_ws[0]

    # reset to None
    if len(opt.images) == 0:
        opt.images = None

    # default parameters for finetuning
    if opt.dmtet:
        opt.h = int(opt.h * opt.dmtet_reso_scale)
        opt.w = int(opt.w * opt.dmtet_reso_scale)

        opt.t_range = [0.02, 0.50] # ref: magic3D

        # assume finetuning
        opt.warmup_iters = 0
        opt.progressive_view = False
        opt.progressive_level = False

        if opt.guidance != 'zero123':
            # smaller fovy (zoom in) for better details
            opt.fovy_range = [opt.fovy_range[0] - 10, opt.fovy_range[1] - 10]

    # record full range for progressive view expansion
    if opt.progressive_view:
        # disable as they disturb progressive view
        opt.jitter_pose = False
        opt.uniform_sphere_rate = 0
        # back up full range
        opt.full_radius_range = opt.radius_range
        opt.full_theta_range = opt.theta_range
        opt.full_phi_range = opt.phi_range
        opt.full_fovy_range = opt.fovy_range

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
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt).to(device)

    if opt.dmtet and opt.init_ckpt != '':
        # load pretrained weights to init dmtet
        state_dict = torch.load(opt.init_ckpt, map_location=device)
        model.load_state_dict(state_dict['model'], strict=False)
        if opt.cuda_ray:
            model.mean_density = state_dict['mean_density']
        model.init_tet()

    print(model)

    if opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader()
            trainer.test(test_loader)

            if opt.save_mesh:
                trainer.save_mesh()

    else:

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train).dataloader()

        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else: # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.backbone == 'vanilla':
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
            # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        if opt.guidance == 'stable-diffusion':
            from guidance.sd_utils import StableDiffusion
            guidance = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range)
        elif opt.guidance == 'zero123':
            from guidance.zero123_utils import Zero123
            guidance = Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, zero123_final=opt.zero123_final)
        elif opt.guidance == 'clip':
            from guidance.clip_utils import CLIP
            guidance = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        trainer.default_view_data = train_loader._data.get_default_view_data()

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader()
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            epoch_freq = np.ceil((opt.test_freq or opt.iters) / len(train_loader)).astype(np.int32)
            max_epochs = np.arange(trainer.epoch, max_epoch, epoch_freq) + epoch_freq

            for max_epoch in max_epochs:
                trainer.train(train_loader, valid_loader, max_epoch)

                # also test at the end
                trainer.test(test_loader)

                if opt.save_mesh:
                    trainer.save_mesh()

