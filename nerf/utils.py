import os
import glob
import tqdm
import random
import logging
import gc 

import numpy as np
import imageio, imageio_ffmpeg 
import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torchmetrics.functional import pearson_corrcoef

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

from nerf.clip import CLIP
from easydict import EasyDict as edict
logger = logging.getLogger(__name__)


class AverageMeters(object):
    """Computes and stores the average and current value"""
    def __init__(self, keys=['loss']):
        self.meters = edict()
        self.keys= keys
        self.reset()

    def reset(self):
        for key in self.keys:
            self.meters[key] = {}
            self.meters[key].val = 0
            self.meters[key].avg = 0
            self.meters[key].sum = 0
            self.meters[key].count = 0

    def reset_by_key(self, key):
        self.meters[key] = {}
        self.meters[key].val = 0
        self.meters[key].avg = 0
        self.meters[key].sum = 0
        self.meters[key].count = 0

    def update(self, in_dict, n=1):
        for key, val in in_dict.items():
            if key not in self.keys:
                self.keys.append(key)
                self.reset_by_key(key)
            self.meters[key].val = val
            self.meters[key].sum += val * n
            self.meters[key].count += n
            self.meters[key].avg = self.meters[key].sum / self.meters[key].count


def setup_workspace(opt):
    if opt.workspace is None or opt.workspace == '':
        opt.workspace = 'out/'
        if opt.text:
            opt.workspace += '_'.join(opt.text.split(' '))
        if opt.image:
            opt.workspace += '_'.join('_'.join(opt.image.split('/')
                                      [-2:]).split('.')[:-1])
        opt.workspace += '+' + time.strftime('%Y%m%d-%H%M%S')
    opt.runname = os.path.basename(opt.workspace)
    os.makedirs(opt.workspace, exist_ok=True)
    opt.log_path = os.path.join(opt.workspace, f"log_{opt.runname}.txt")
    opt.ckpt_path = os.path.join(opt.workspace, 'checkpoints')
    opt.best_path = f"{opt.ckpt_path}/{opt.runname}.pth"
    os.makedirs(opt.ckpt_path, exist_ok=True)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def save_tensor2image(x: torch.Tensor, path, channel_last=False, quality=75, **kwargs):
    # assume the input x is channel last
    if x.ndim == 4 and channel_last:
        x = x.permute(0, 3, 1, 2) 
    TF.to_pil_image(make_grid(x, value_range=(0, 1), **kwargs)).save(path, quality=quality)

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def nonzero_normalize_depth(depth, mask=None):
    if mask is not None:
        if (depth[mask]>0).sum() > 0:
            nonzero_depth_min = depth[mask][depth[mask]>0].min()
        else:
            nonzero_depth_min = 0
    else:
        if (depth>0).sum() > 0:
            nonzero_depth_min = depth[depth>0].min()
        else:
            nonzero_depth_min = 0
    if nonzero_depth_min == 0:
        return depth
    else:
        depth = (depth - nonzero_depth_min) / depth.max()
        return depth.clamp(0, 1)


class Trainer(object):
    def __init__(self,
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 max_keep_ckpt=1, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboard=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):

        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.use_checkpoint = use_checkpoint
        self.use_tensorboard = use_tensorboard
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance
        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}
            self.prepare_embeddings()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.novel_view_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }
        self.loss_meter = AverageMeters()
        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        logger.info(f'[INFO] cmdline: {self.argv}')
        logger.info(f'args:\n{self.opt}')
        logger.info(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        logger.info(
            f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(f'[INFO] #Optimizer: \n{self.optimizer}')
        logger.info(f'[INFO] #Scheduler: \n{self.lr_scheduler}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                logger.info("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                logger.info("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                logger.info("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.opt.best_path):
                    logger.info("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.opt.best_path)
                else:
                    logger.info(
                        f"[INFO] {self.opt.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                logger.info(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self):

        # text embeddings (stable-diffusion)
        if self.opt.text is not None:

            dir_texts = ['front', 'side', 'back']
            if 'SD' in self.guidance:
                self.embeddings['SD']['default'] = self.guidance['SD'].get_all_text_embeds([self.opt.text])
                neg_embedding = self.guidance['SD'].get_all_text_embeds([self.opt.negative])

                for idx, d in enumerate(dir_texts):
                    text = f"{self.opt.text}, {d} view"
                    self.embeddings['SD'][d] = self.guidance['SD'].get_all_text_embeds([text])
                    if self.opt.dir_texts_neg:
                        text_neg = self.opt.negative + ', '.join([text+' view' for i, text in enumerate(dir_texts) if i != idx]) 
                        logger.info(f'dir_texts of {d}\n postive text: {text},\n negative text: {text_neg}')
                        neg_embedding= self.guidance['SD'].get_all_text_embeds([text_neg])
                    self.embeddings['SD'][d] = torch.cat((neg_embedding, self.embeddings['SD'][d]), dim=0)

            if 'IF' in self.guidance:
                self.embeddings['IF']['default'] = self.guidance['IF'].get_text_embeds([self.opt.text])
                neg_embedding = self.guidance['IF'].get_text_embeds([self.opt.negative])

                for idx, d in enumerate(dir_texts):
                    text = f"{self.opt.text}, {d} view"
                    self.embeddings['IF'][d] = self.guidance['IF'].get_text_embeds([text])
                    if self.opt.dir_texts_neg:
                        text_neg = self.opt.negative + ', '.join([text+' view' for i, text in enumerate(dir_texts) if i != idx]) 
                        logger.info(f'dir_texts of {d}\n postive text: {text},\n negative text: {text_neg}')
                        neg_embedding= self.guidance['IF'].get_all_text_embeds([text_neg])
                    self.embeddings['IF'][d] = torch.cat((neg_embedding, self.embeddings['IF'][d]), dim=0)
                
            # if 'clip' in self.guidance:
            #     self.embeddings['clip']['text'] = self.guidance['clip'].get_text_embeds(self.opt.text)

        if self.opt.images is not None:

            h = int(self.opt.known_view_scale * self.opt.h)
            w = int(self.opt.known_view_scale * self.opt.w)

            # load processed image and remove edges
            rgbas = []
            rgbas_hw = []
            mask_no_edges = []
            for image in self.opt.images:
                rgba = cv2.cvtColor(cv2.imread(image, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                rgbas.append(rgba)
                rgba_hw = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
                rgbas_hw.append(rgba_hw)
                if self.opt.rm_edge:
                    alpha = np.uint8(rgba_hw[..., 3] * 255.)
                    dilate = cv2.dilate(alpha, np.ones((self.opt.edge_width, self.opt.edge_width), np.uint8))
                    edge = cv2.absdiff(alpha, dilate).astype(np.float32) / 255
                    mask_no_edge = rgba_hw[..., 3] > 0.5
                    mask_no_edge[edge>self.opt.edge_threshold] = False
                    mask_no_edges.append(mask_no_edge)
            rgba_hw = np.stack(rgbas_hw)
            mask = rgba_hw[..., 3] > 0.5
            if len(mask_no_edges) > 0:
                mask_no_edge = np.stack(mask_no_edges)
            else:
                mask_no_edge = mask
                
            # breakpoint() 
            # rgb
            rgb_hw = rgba_hw[..., :3] * rgba_hw[..., 3:] + (1 - rgba_hw[..., 3:]) 
            self.rgb = torch.from_numpy(rgb_hw).permute(0,3,1,2).contiguous().to(self.device)
            self.mask = torch.from_numpy(mask).to(self.device)
            self.opacity = torch.from_numpy(mask_no_edge).to(self.device).to(torch.float32).unsqueeze(0)
            print(f'[INFO] dataset: load image prompt {self.opt.images} {self.rgb.shape}')

            # load depth
            depth_paths = [image.replace('rgba', 'depth') for image in self.opt.images]
            if os.path.exists(depth_paths[0]):
                depths = [cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]
                depth = np.stack([cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA) for depth in depths])
                self.depth = 1 - torch.from_numpy(depth.astype(np.float32) / 255).to(self.device)
                if len(self.depth.shape) == 4 and self.depth.shape[-1] > 1:
                    self.depth = self.depth[..., 0]
                    logger.info(f'[WARN] dataset: {depth_paths[0]} has more than one channel, only use the first channel')
                if self.opt.normalize_depth:
                    self.depth = nonzero_normalize_depth(self.depth, self.mask)
                save_tensor2image(self.depth, os.path.join(self.workspace, 'depth_resized.jpg'))
                self.depth = self.depth[self.mask]
                print(f'[INFO] dataset: load depth prompt {depth_paths} {self.depth.shape}')
            else:
                self.depth = None
                logger.info(f'[WARN] dataset: {depth_paths[0]} is not found')
                
            # load normal
            normal_paths = [image.replace('rgba', 'normal') for image in self.opt.images]
            if os.path.exists(normal_paths[0]):
                normals = []
                for normal_path in normal_paths:
                    normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
                    if normal.shape[-1] == 4:
                        normal = cv2.cvtColor(normal, cv2.COLOR_BGRA2RGB)
                    normals.append(normal)
                normal = np.stack([cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA) for normal in normals])
                self.normal = torch.from_numpy(normal.astype(np.float32) / 255).to(self.device)
                save_tensor2image(self.normal, os.path.join(self.workspace, 'normal_resized.jpg'), channel_last=True)
                print(f'[INFO] dataset: load normal prompt {normal_paths} {self.normal.shape}')
                self.normal = self.normal[self.mask]
            else:
                self.normal = None
                logger.info(f'[WARN] dataset: {normal_paths[0]} is not found')

            # save for debug
            save_tensor2image(self.rgb, os.path.join(self.workspace, 'rgb_resized.png'), channel_last=False)
            save_tensor2image(self.opacity, os.path.join(self.workspace, 'opacity_resized.png'), channel_last=False)

            # encode embeddings for zero123
            if 'zero123' in self.guidance:
                rgba_256 = np.stack([cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
                rgbs_256 = rgba_256[..., :3] * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
                rgb_256 = torch.from_numpy(rgbs_256).permute(0,3,1,2).contiguous().to(self.device)
                guidance_embeds = self.guidance['zero123'].get_img_embeds(rgb_256)
                self.embeddings['zero123']['default'] = {
                    'zero123_ws' : self.opt.zero123_ws,
                    'c_crossattn' : guidance_embeds[0],
                    'c_concat' : guidance_embeds[1],
                    'ref_polars' : self.opt.ref_polars,
                    'ref_azimuths' : self.opt.ref_azimuths,
                    'ref_radii' : self.opt.ref_radii,
                }

            # if 'clip' in self.guidance:
            #     self.embeddings['clip']['image'] = self.guidance['clip'].get_img_embeds(self.rgb)
                        # encoder image for clip
            if self.opt.use_clip:
                self.rgb_clip_embed = self.guidance.get_clip_img_embeds(self.rgb)
                # debug.
                scaler = torch.cuda.amp.GradScaler()
                image = torch.randn((1,3,512,512), device=self.device, requires_grad=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = self.guidance.clip_loss(self.rgb_clip_embed, image)
                scaler.scale(loss).backward()
            else:
                self.rgb_clip_embed = None


    # ------------------------------
    @torch.no_grad()
    def match_known(self, **kwargs):
        self.model.eval()
        data = self.default_view_data
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        mvp = data['mvp']  # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        ambient_ratio = 1.0
        shading = self.opt.known_shading
        binarize = False
        bg_color = self.get_bg_color(
            self.opt.bg_color_known, B*N, rays_o.device)

        # add camera noise to avoid grid-like artifect
        # * (1 - self.global_step / self.opt.iters)
        noise_scale = self.opt.known_view_noise_scale
        rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True,
                                    bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(
            0, 3, 1, 2).contiguous()  # [1, 3, H, W]
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)

        rgb_loss = self.opt.lambda_rgb * \
            F.mse_loss(pred_rgb*self.opacity,
                       self.rgb*self.opacity)
        mask_loss = self.opt.lambda_mask * \
            F.mse_loss(pred_mask, self.mask.to(torch.float32).unsqueeze(0))
        return pred_rgb, pred_mask, rgb_loss, mask_loss

    def get_bg_color(self, bg_type, N, device):
        if bg_type is None:
            return None
        elif isinstance(bg_type, str):
            if bg_type == 'pixelnoise':
                bg_color = torch.rand((N, 3), device=device)
            elif bg_type == 'noise':
                bg_color = torch.rand((1, 3), device=device).repeat(N, 1)
            elif bg_type == 'white':
                bg_color = torch.ones((N, 3), device=device)
            return bg_color
        elif isinstance(bg_type, Tensor):
            bg_color = bg_color.to(device)
            return bg_color
        else:
            raise NotImplementedError(f"{bg_type} is not implemented")

    # def margin_rank_loss(self, depth):
    #     # high res, only calc on fg
    #     output = depth.squeeze().view(-1)
    #     output = output[self.fg_idx]
    #     num = output.shape[0] # [n, 1]
    #     # print(num)
    #     output = output.reshape(1, -1)
    #     o1 = output.expand(num, -1).reshape(-1)
    #     o2 = output.T.expand(-1, num).reshape(-1)
    #     return F.margin_ranking_loss(o1, o2, self.rank_loss_target)

    def train_step(self, data):
        # perform RGBD loss instead of SDS if is image-conditioned
        do_rgbd_loss = self.opt.images is not None and \
            (self.global_step < self.opt.known_iters) or (self.global_step % self.opt.known_view_interval == 0)

        # override random camera with fixed known camera
        if do_rgbd_loss:
            data = self.default_view_data

        # progressively relaxing view range
        if self.opt.progressive_view:
            r = min(1.0, 0.2 + self.global_step / (0.5 * self.opt.iters))
            self.opt.phi_range = [self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                    self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                    self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]

        # progressively increase max_level
        if self.opt.progressive_level:
            self.model.max_level = min(1.0, 0.25 + self.global_step / (0.5 * self.opt.iters))

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp'] # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # When ref_data has B images > opt.batch_size
        if B > self.opt.batch_size:
            # choose batch_size images out of those B images
            choice = torch.randperm(B)[:self.opt.batch_size]
            B = self.opt.batch_size
            rays_o = rays_o[choice]
            rays_d = rays_d[choice]
            mvp = mvp[choice]

        if do_rgbd_loss:
            ambient_ratio = 1.0
            shading = 'lambertian' # use lambertian instead of albedo to get normal
            as_latent = False
            binarize = False
            bg_color = self.get_bg_color(
                self.opt.bg_color_known, B*N, rays_o.device)

            # add camera noise to avoid grid-like artifact
            if self.opt.known_view_noise_scale > 0:
                noise_scale = self.opt.known_view_noise_scale #* (1 - self.global_step / self.opt.iters)
                rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
                rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        elif self.global_step < (self.opt.latent_iter_ratio * self.opt.iters):
            ambient_ratio = 1.0
            shading = 'normal'
            as_latent = True
            binarize = False
            bg_color = None

        else:
            if self.global_step < (self.opt.normal_iter_ratio * self.opt.iters):
                ambient_ratio = 1.0
                shading = 'normal'
            elif self.global_step < (self.opt.textureless_iter_ratio * self.opt.iters):
                ambient_ratio = 0.1 + 0.9 * random.random()
                shading = 'textureless'
            elif self.global_step < (self.opt.albedo_iter_ratio * self.opt.iters):
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                ambient_ratio = 0.1 + 0.9 * random.random()
                rand = random.random()
                if rand > 0.8:
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

            as_latent = False

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            if self.opt.bg_radius > 0 and rand > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        if self.opt.normalize_depth: 
            pred_depth = nonzero_normalize_depth(pred_depth)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
        else:
            pred_normal = None 

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous() # [B, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
        out_dict = {
            'rgb': pred_rgb,
            'depth': pred_depth,
            'mask': pred_mask,
            'normal': pred_normal,
        }

        # Loss
        # known view loss
        loss_rgb, loss_mask, loss_normal, loss_depth, loss_sds, loss_if, loss_zero123, loss_clip, loss_entropy, loss_opacity, loss_orient, loss_smooth, loss_smooth2d, loss_smooth3d, loss_mesh_normal, loss_mesh_lap = torch.zeros(16, device=self.device)
        # known view loss
        if do_rgbd_loss:
            gt_mask = self.mask # [B, H, W]
            gt_rgb = self.rgb   # [B, 3, H, W]
            gt_opacity = self.opacity   # [B, 1, H, W]
            gt_normal = self.normal # [B, H, W, 3]
            gt_depth = self.depth   # [B, H, W]

            if len(gt_rgb) > self.opt.batch_size:
                gt_mask = gt_mask[choice]
                gt_rgb = gt_rgb[choice]
                gt_opacity = gt_opacity[choice]
                gt_normal = gt_normal[choice]
                gt_depth = gt_depth[choice]

            # color loss
            loss_rgb = self.opt.lambda_rgb * \
                F.mse_loss(pred_rgb*gt_opacity, gt_rgb*gt_opacity)

            # mask loss
            loss_mask = self.opt.lambda_mask * F.mse_loss(pred_mask, gt_mask.to(torch.float32).unsqueeze(0))

            # normal loss
            if self.opt.lambda_normal > 0 and 'normal_image' in outputs and self.normal is not None:
                pred_normal = pred_normal[self.mask]
                lambda_normal = self.opt.lambda_normal * \
                    min(1, self.global_step / self.opt.iters)                
                loss_normal = lambda_normal * \
                    (1 - F.cosine_similarity(pred_normal, self.normal).mean())/2

            # relative depth loss
            if self.opt.lambda_depth > 0 and self.depth is not None:
                valid_pred_depth = pred_depth[:, 0][self.mask]
                loss_depth = self.opt.lambda_depth * (1 - pearson_corrcoef(valid_pred_depth, self.depth))/2
            
            loss = loss_rgb + loss_mask + loss_normal + loss_depth
        # novel view loss
        else:
            save_guidance_path = os.path.join(self.opt.workspace, 'guidance', f'train_step{self.global_step}_guidance.jpg') if self.opt.save_guidance_every > 0 and self.novel_view_step % self.opt.save_guidance_every ==0 else None
            if 'SD' in self.guidance:
                # interpolate text_z
                azimuth = data['azimuth'] # [-180, 180]

                # ENHANCE: remove loop to handle batch size > 1
                text_z = [] 
                for b in range(azimuth.shape[0]):
                    if azimuth[b] >= -90 and azimuth[b] < 90:
                        if azimuth[b] >= 0:
                            r = 1 - azimuth[b] / 90
                        else:
                            r = 1 + azimuth[b] / 90
                        start_z = self.embeddings['SD']['front']
                        end_z = self.embeddings['SD']['side']
                    else:
                        if azimuth[b] >= 0:
                            r = 1 - (azimuth[b] - 90) / 90
                        else:
                            r = 1 + (azimuth[b] + 90) / 90
                        start_z = self.embeddings['SD']['side']
                        end_z = self.embeddings['SD']['back']
                    text_z.append(r * start_z + (1 - r) * end_z)
                text_z = torch.stack(text_z, dim=0).transpose(0, 1).flatten(0, 1)
                text_z_sds = text_z[:, :-1] 
                loss_sds, _ = self.guidance['SD'].train_step(text_z_sds, pred_rgb, as_latent=as_latent, guidance_scale=self.opt.guidance_scale['SD'], grad_scale=self.opt.lambda_guidance['SD'],
                                                             density=pred_mask if self.opt.gudiance_spatial_weighting else None, 
                                                             save_guidance_path=save_guidance_path
                                                             )
                # if self.opt.lambda_clip > 0:
                #     lambda_clip = 10 * (1 - abs(azimuth) / 180) * self.opt.lambda_clip
                #     if self.opt.clip_image_loss:
                #         loss_clip = lambda_clip * self.guidance.clip_loss(self.rgb_clip_embed, pred_rgb)
                #     else:
                #         loss_clip = lambda_clip * self.guidance.clip_loss(text_z_clip, pred_rgb)
                        
            if 'IF' in self.guidance:
                # interpolate text_z
                azimuth = data['azimuth'] # [-180, 180]

                # ENHANCE: remove loop to handle batch size > 1
                # ENHANCE: remove loop to handle batch size > 1
                text_z = [] 
                for b in range(azimuth.shape[0]):
                    if azimuth[b] >= -90 and azimuth[b] < 90:
                        if azimuth[b] >= 0:
                            r = 1 - azimuth[b] / 90
                        else:
                            r = 1 + azimuth[b] / 90
                        start_z = self.embeddings['IF']['front']
                        end_z = self.embeddings['IF']['side']
                    else:
                        if azimuth[b] >= 0:
                            r = 1 - (azimuth[b] - 90) / 90
                        else:
                            r = 1 + (azimuth[b] + 90) / 90
                        start_z = self.embeddings['IF']['side']
                        end_z = self.embeddings['IF']['back']
                    text_z.append(r * start_z + (1 - r) * end_z)
                text_z = torch.stack(text_z, dim=0).transpose(0, 1).flatten(0, 1)
                text_z = torch.cat(text_z, dim=1).reshape(B, 2, start_z.shape[-2]-1, start_z.shape[-1]).transpose(0, 1).flatten(0, 1)
                loss_if = self.guidance['IF'].train_step(text_z, pred_rgb, guidance_scale=self.opt.guidance_scale['IF'], grad_scale=self.opt.lambda_guidance['IF'])

            if 'zero123' in self.guidance:

                polar = data['polar']
                azimuth = data['azimuth']
                radius = data['radius']

                loss_zero123 = self.guidance['zero123'].train_step(self.embeddings['zero123']['default'], pred_rgb, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale['zero123'],
                                                                  as_latent=as_latent, grad_scale=self.opt.lambda_guidance['zero123'], save_guidance_path=save_guidance_path)

            if 'clip' in self.guidance:

                # empirical, far view should apply smaller CLIP loss
                lambda_guidance = 10 * (1 - abs(azimuth) / 180) * self.opt.lambda_guidance['clip']
                loss_clip = self.guidance['clip'].train_step(self.embeddings['clip'], pred_rgb, grad_scale=lambda_guidance)
            loss = loss_sds + loss_if + loss_zero123 + loss_clip

        # regularizations
        if not self.opt.dmtet:

            if self.opt.lambda_opacity > 0:
                loss_opacity = self.opt.lambda_opacity * (outputs['weights_sum'] ** 2).mean()

            if self.opt.lambda_entropy > 0:
                lambda_entropy = self.opt.lambda_entropy * \
                    min(1, 2 * self.global_step / self.opt.iters)
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = lambda_entropy * (- alphas * torch.log2(alphas) -
                                (1 - alphas) * torch.log2(1 - alphas)).mean()

            if self.opt.lambda_normal_smooth > 0 and 'normal_image' in outputs:
                pred_vals = outputs['normal_image'].reshape(B, H, W, 3)
                # total-variation
                loss_smooth = (pred_vals[:, 1:, :, :] - pred_vals[:, :-1, :, :]).square().mean() + \
                              (pred_vals[:, :, 1:, :] -
                               pred_vals[:, :, :-1, :]).square().mean()
                loss_smooth = self.opt.lambda_normal_smooth * loss_smooth

            if self.opt.lambda_normal_smooth2d > 0 and 'normal_image' in outputs:
                pred_vals = outputs['normal_image'].reshape(
                    B, H, W, 3).permute(0, 3, 1, 2).contiguous()
                smoothed_vals = TF.gaussian_blur(pred_vals, kernel_size=9)
                loss_smooth2d = self.opt.lambda_normal_smooth2d * F.mse_loss(pred_vals, smoothed_vals)

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = self.opt.lambda_orient * outputs['loss_orient']
            
            if self.opt.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs:
                loss_smooth3d = self.opt.lambda_3d_normal_smooth * outputs['loss_normal_perturb']

            loss += loss_opacity + loss_entropy + loss_smooth + loss_smooth2d + loss_orient + loss_smooth3d
            
        else:
            if self.opt.lambda_mesh_normal > 0:
                loss_mesh_normal = self.opt.lambda_mesh_normal * \
                    outputs['loss_normal']

            if self.opt.lambda_mesh_lap > 0:
                loss_mesh_lap = self.opt.lambda_mesh_lap * outputs['loss_lap']
            loss += loss_mesh_normal + loss_mesh_lap

        losses_dict = {
                    'loss': loss.item(),
                    'loss_sds': loss_sds.item(),
                    'loss_if': loss_if.item(),
                    'loss_zero123': loss_zero123.item(),
                    'loss_clip': loss_clip.item(),
                    'loss_rgb': loss_rgb.item(),
                    'loss_mask': loss_mask.item(),
                    'loss_normal': loss_normal.item(),
                    'loss_depth': loss_depth.item(),
                    'loss_opacity': loss_opacity.item(),
                    'loss_entropy': loss_entropy.item(),
                    'loss_smooth': loss_smooth.item(),
                    'loss_smooth2d': loss_smooth2d.item(),
                    'loss_smooth3d': loss_smooth3d.item(),
                    'loss_orient': loss_orient.item(),
                    'loss_mesh_normal': loss_mesh_normal.item(),
                    'loss_mesh_lap': loss_mesh_lap.item(),
                }
        # if loss_guidance_dict:
        #     for key, val in loss_guidance_dict.items():
        #         losses_dict[key] = val.item() if isinstance(val, torch.Tensor) else val
                
        if 'normal' in out_dict:
            out_dict['normal'] = out_dict['normal'].permute(0, 3, 1, 2).contiguous()

        # save for debug purpose
        if self.opt.save_train_every > 0 and self.global_step % self.opt.save_train_every == 0:
            image_save_path = os.path.join(self.workspace, 'train_debug',)
            os.makedirs(image_save_path, exist_ok=True)
            for key, value in out_dict.items():
                if value is not None:
                    value = ((value - value.min()) / (value.max() - value.min() + 1e-6)).detach().mul(255).to(torch.uint8)
                    try:
                        save_tensor2image(value, os.path.join(image_save_path, f'train_{self.global_step:06d}_{key}.jpg'), channel_last=False) 
                    except:
                        pass
        return loss, losses_dict, out_dict 

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':

            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'lambertian' 
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W, 1)
        if self.opt.normalize_depth: 
            pred_depth = nonzero_normalize_depth(pred_depth)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
        else:
            pred_normal = None 
        out_dict = {
            shading: pred_rgb,
            'depth': pred_depth,
            'normal_image': pred_normal,
        }
        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)
        return out_dict, loss

    def test_step(self, data, bg_color=None, perturb=False, shading='lambertian'):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        bg_color = self.get_bg_color(bg_color, B*N, rays_o.device)

        shading = data['shading'] if 'shading' in data else shading 
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W, 1)
        pred_mask = outputs['weights_sum'].reshape(B, H, W, 1)
        # if self.opt.normalize_depth: 
        pred_depth = nonzero_normalize_depth(pred_depth)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
            pred_normal = pred_normal * pred_mask + (1.0 - pred_mask) 
        else:
            pred_normal = None 
        out_dict = {
            shading: pred_rgb,
            'depth': pred_depth,
            'normal_image': pred_normal,
            'mask': pred_mask,
        }
        return out_dict

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        logger.info(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)

        logger.info(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, test_loader, max_epochs):

        if self.use_tensorboard and self.local_rank == 0:
            self.writer = SummaryWriter(
                os.path.join(self.workspace, "run", self.name))

        # init from nerf should be performed after Shap-E, since Shap-E will rescale dmtet
        if self.opt.dmtet and (self.opt.init_ckpt and os.path.exists(self.opt.init_ckpt)):
            reset_scale = False if self.opt.use_shape else True
            old_sdf = self.model.get_sdf_from_nerf(reset_scale)
            if not self.opt.tet_mlp:
                self.model.dmtet.init_tet_from_sdf(old_sdf)
                self.test(valid_loader, name=f'init_ckpt', write_video=False, save_each_frame=False, subfolder='check_init')
        else:
            old_sdf = None
            
        if self.opt.use_shape and self.opt.dmtet:
            os.makedirs(os.path.join(self.opt.workspace, 'shape'), exist_ok=True)
            best_loss = torch.inf
            best_idx = 0
            for idx, (sdf, color) in enumerate(zip(self.opt.rpsts, self.opt.colors)):
                self.model.init_tet_from_sdf_color(sdf)
                pred_rgb, pred_mask, rgb_loss, mask_loss = self.match_known()
                best_loss = min(best_loss, mask_loss)
                if best_loss == mask_loss:
                    best_idx = idx
                    logger.info(f"==> Current best match shape known sdf idx: {best_idx}")
                save_tensor2image(pred_mask, os.path.join(self.opt.workspace, 'shape', f"match_shape_known_{idx}_rgb.jpg"), channel_last=False)
                self.test(valid_loader, name=f'idx_{idx}', write_video=False, save_each_frame=False, subfolder='check_init')
              
            sdf = self.opt.rpsts[best_idx]
            self.model.init_tet_from_sdf_color(sdf, self.opt.colors[best_idx])
            self.test(valid_loader, name=f'shape_only', write_video=False, save_each_frame=False, subfolder='check_init')

            # Enable mixture model
            if self.opt.base_mesh:
                logger.info(f"==> Enable mixture model with base mesh {self.opt.base_mesh}")
                mesh_sdf = self.model.dmtet.get_sdf_from_mesh(self.opt.base_mesh)
                sdf = (mesh_sdf.clamp(0, 1) + sdf.clamp(0,1) ).clamp(0, 1)

            if old_sdf is not None:
                sdf = (sdf.clamp(0, 1) + old_sdf.clamp(0, 1)).clamp(0, 1)

            self.model.init_tet_from_sdf_color(sdf, self.opt.colors[best_idx])
            self.test(valid_loader, name=f'shape_merge', write_video=False, save_each_frame=False, subfolder='check_init')

            del best_loss, best_idx, pred_rgb, pred_mask, rgb_loss, mask_loss
            self.opt.rpsts = None
            gc.collect()
            torch.cuda.empty_cache()

        # init shape for NeRF. NOTE: Does not work yet.. in progress.
        # if self.opt.use_shape and not self.opt.dmtet:
        #     os.makedirs(os.path.join(self.opt.workspace, 'shape'), exist_ok=True)
        #     best_loss = torch.inf
        #     best_idx = 0
        #     for idx, (density, color) in enumerate(zip(self.opt.rpsts, self.opt.colors)):
        #         self.model.init_nerf_from_sdf_color(density, color, self.opt.points, lr=0.001)
        #         pred_rgb, pred_mask, rgb_loss, mask_loss = self.match_known()
        #         best_loss = min(best_loss, mask_loss)
        #         if best_loss == mask_loss:
        #             best_idx = idx
        #             logger.info(f"==> Current best match shape known sdf idx: {best_idx}")
        #         save_tensor2image(pred_mask, os.path.join(self.opt.workspace, 'shape', f"match_shape_known_{idx}_rgb.jpg"), channel_last=False)
        #         self.evaluate_one_epoch(valid_loader, f'idx_{idx}')
        #     self.model.init_nerf_from_sdf_color(self.opt.rpsts[best_idx], self.opt.colors[best_idx])
        #     self.evaluate_one_epoch(valid_loader, f'init_from_shape_{idx}')

        #     del best_loss, best_idx, pred_rgb, pred_mask, rgb_loss, mask_loss
        #     self.opt.rpsts = None
        #     self.opt.colors = None
        #     self.opt.points = None
        #     gc.collect()
        #     torch.cuda.empty_cache()

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader, max_epochs)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            if self.epoch % self.opt.test_interval == 0 or self.epoch == max_epochs:
                self.test(test_loader, img_folder='images' if self.epoch == max_epochs else f'images_ep{self.epoch:04d}')

        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        logger.info(f"[INFO] training takes {(self.total_train_t)/ 60:.4f} minutes.")

        if self.use_tensorboard and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboard, use_tensorboard = False, self.use_tensorboard
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboard = use_tensorboard

    def test(self, loader, save_path=None, name=None, 
             write_video=True, save_each_frame=True, shading='lambertian', 
             subfolder='results', img_folder='images'
             ):

        if save_path is None:
            save_path = os.path.join(self.workspace, subfolder)
        image_save_path = os.path.join(self.workspace, subfolder, img_folder)

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(image_save_path, exist_ok=True)

        logger.info(f"==> Start Test, saving {shading} results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_outputs = {} 
        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.test_step(data, bg_color=self.opt.bg_color_test, shading=shading)
                for key, value in outputs.items():
                    if value is not None:
                        value = ((value - value.min()) / (value.max() - value.min() + 1e-6)).detach().mul(255).to(torch.uint8)
                        if save_each_frame:
                            save_tensor2image(value, os.path.join(image_save_path, f'{name}_{i:04d}_{key}.jpg'), channel_last=True) 
                        if key not in all_outputs.keys():
                            all_outputs[key] = []
                        all_outputs[key].append(value)
                pbar.update(loader.batch_size)

        for key, value in all_outputs.items():
            all_outputs[key] = torch.cat(value, dim=0)
            
        if write_video:
            for key, value in all_outputs.items():
                # current version torchvision does not support writing a single-channel video
                # torchvision.io.write_video(os.path.join(save_path, f'{name}_{key}.mp4'), all_outputs[key].detach().cpu(), fps=25)
                imageio.mimwrite(os.path.join(save_path, f'{name}_{key}.mp4'), all_outputs[key].detach().cpu().numpy(), fps=25, quality=8, macro_block_size=1)
        for key, value in all_outputs.items():
            save_tensor2image(value, os.path.join(save_path, f'{name}_{key}.jpg'), channel_last=True)
        logger.info(f"==> Finished Test.")

    # [GUI] train text step.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                loss, loss_dicts, outputs = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            self.loss_meter.update(loss_dicts)
            
        if self.ema is not None:
            self.ema.update()

        average_loss = self.loss_meter.meters['loss'].avg

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs


    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, mvp, W, H, bg_color=None, spp=1, downscale=1, light_d=None, ambient_ratio=1.0, shading='albedo'):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        mvp = torch.from_numpy(mvp).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        # from degree theta/phi to 3D normalized vec
        light_d = np.deg2rad(light_d)
        light_d = np.array([
            np.sin(light_d[0]) * np.sin(light_d[1]),
            np.cos(light_d[0]),
            np.sin(light_d[0]) * np.cos(light_d[1]),
        ], dtype=np.float32)
        light_d = torch.from_numpy(light_d).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'mvp': mvp,
            'H': rH,
            'W': rW,
            'light_d': light_d,
            'ambient_ratio': ambient_ratio,
            'shading': shading,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                outputs = self.test_step(
                    data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # have to permute twice with torch...
            outputs[shading] = F.interpolate(outputs[shading].permute(0, 3, 1, 2), size=(
                H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            outputs['depth'] = F.interpolate(outputs['depth'].unsqueeze(
                1), size=(H, W), mode='nearest').squeeze(1)

            if outputs['normal_imagea'] is not None:
                outputs['normal_image'] = F.interpolate(outputs['normal_image'].unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        return outputs

    def train_one_epoch(self, loader, max_epochs):
        logger.info(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        self.local_step = 0

        for data in loader:

            # update grid every 16 steps
            if (self.model.cuda_ray or self.model.taichi_ray) and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            # Update grid
            if self.opt.grid_levels_mask > 0:
                if self.global_step > self.opt.grid_levels_mask_iters:
                    self.model.grid_levels_mask = 0
                else:
                    self.model.grid_levels_mask = self.opt.grid_levels_mask

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                loss, losses_dict, outputs = self.train_step(data)

            # hooked grad clipping for RGB space
            if self.opt.grad_clip_rgb >= 0:
                def _hook(grad):
                    if self.opt.fp16:
                        # correctly handle the scale
                        grad_scale = self.scaler._get_scale_async()
                        return grad.clamp(grad_scale * -self.opt.grad_clip_rgb, grad_scale * self.opt.grad_clip_rgb)
                    else:
                        return grad.clamp(-self.opt.grad_clip_rgb, self.opt.grad_clip_rgb)
                outputs['rgb'].register_hook(_hook)
                # if (self.global_step <= self.opt.known_iters or self.global_step % self.opt.known_view_interval == 0) and self.opt.image is not None and self.opt.joint_known_unknown and known_rgbs is not None:
                #     known_rgbs.register_hook(_hook)
                # pred_rgbs.retain_grad()

            self.scaler.scale(loss).backward()

            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            
            self.loss_meter.update(losses_dict)
            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:
                #         metric.update(preds, truths)

                if self.use_tensorboard:

                    for key, val in losses_dict.items():
                        self.writer.add_scalar(
                            f"train/{key}", val, self.global_step) 

                    self.writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.global_step % self.opt.log_every == 0:
                    strings = f"==> Train [Step] {self.global_step}/{self.opt.iters}"
                    for key, value in losses_dict.items():
                        strings += f", {key}={value:.4f}"
                    logger.info(strings)
                    strings = f"==> Train [Avg] {self.global_step}/{self.opt.iters}"
                    for key in self.loss_meter.meters.keys():
                        strings += f", {key}={self.loss_meter.meters[key].avg:.4f}"
                    logger.info(strings)

        if self.ema is not None:
            self.ema.update()
            
        average_loss = self.loss_meter.meters['loss'].avg
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            # pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    logger.info(metric.report(), style="red")
                    if self.use_tensorboard:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        # Visualize Training
        if self.local_rank == 0:
            # save image
            save_path = os.path.join(
                self.workspace, 'training')
            os.makedirs(save_path, exist_ok=True)
            name = f'train_{self.name}_ep{self.epoch:04d}'
            for key, value in outputs.items():
                save_tensor2image(value, os.path.join(save_path, f'{name}_{key}.jpg'), channel_last=False) 
        gpu_mem = get_GPU_mem()[0]
        logger.info(f"==> [Finished Epoch {self.epoch}/{max_epochs}. GPU={gpu_mem:.1f}GB.")

    def evaluate_one_epoch(self, loader, name=None):
        logger.info(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()
        
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
                
            all_outputs = {}  
            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    for key, value in outputs.items():
                        if value is not None:
                            dist.all_gather(outputs[key])
                            outputs[key] = torch.cat(outputs[key], dim=0)
                            
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # save image
                    save_path = os.path.join(
                        self.workspace, 'validation')

                    # logger.info(f"==> Saving validation image to {save_path}")
                    os.makedirs(save_path, exist_ok=True)

                    for key, value in outputs.items():
                        if value is not None:
                            value = ((value - value.min()) / (value.max() - value.min() + 1e-6)).detach().mul(255).to(torch.uint8)
                            # save_tensor2image(value, os.path.join(save_path, f'{name}_{self.local_step:04d}_{key}.jpg')) 
                            if key not in all_outputs.keys():
                                all_outputs[key] = []
                            all_outputs[key].append(value)

                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                logger.info(metric.report(), style="blue")
                if self.use_tensorboard:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()
                
            for key, value in all_outputs.items():
                all_outputs[key] = torch.cat(value, dim=0)
                save_tensor2image(all_outputs[key], os.path.join(save_path, f'{name}_{key}.jpg'), channel_last=True)
        if self.ema is not None:
            self.ema.restore()

        logger.info(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.dmtet.tet_scale.cpu().numpy()

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(
                    self.opt.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.opt.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # logger.info(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.opt.best_path)
            else:
                logger.info(
                    f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.opt.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                logger.info(
                    "[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            logger.info("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("[INFO] loaded model.")
        if len(missing_keys) > 0:
            logger.info(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.info(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                logger.info("[INFO] loaded EMA.")
            except:
                logger.info("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                new_scale = torch.from_numpy(
                    checkpoint_dict['tet_scale']).to(self.device)
                self.model.dmtet.verts *= new_scale / self.model.dmtet.tet_scale
                self.model.dmtet.tet_scale = new_scale
            # self.model.init_tet() 
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        logger.info(
            f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("[INFO] loaded optimizer.")
            except:
                logger.info("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                logger.info("[INFO] loaded scheduler.")
            except:
                logger.info("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                logger.info("[INFO] loaded scaler.")
            except:
                logger.info("[WARN] Failed to load scaler.")


def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mems.append(int(((mem_total - mem_free)/1024**3)*1000)/1000)
        mem += mems[-1]
    return mem, mems
