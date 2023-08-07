import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from easydict import EasyDict as edict

class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image


def get_rgba(image, alpha_matting=False):
    try:
        from rembg import remove
    except ImportError:
        print('Please install rembg with "pip install rembg"')
        sys.exit()
    return remove(image, alpha_matting=alpha_matting)


class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(
            self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


class DPT():
    def __init__(self, task='depth', device='cuda'):

        self.task = task
        self.device = device

        from dpt import DPTDepthModel

        if task == 'depth':
            path = 'pretrained/omnidata/omnidata_dpt_depth_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384')
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])

        else: # normal
            path = 'pretrained/omnidata/omnidata_dpt_normal_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor()
            ])

        # load model
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

        
    @torch.no_grad()
    def __call__(self, image):
        # image: np.ndarray, uint8, [H, W, 3]
        H, W = image.shape[:2]
        image = Image.fromarray(image)

        image = self.aug(image).unsqueeze(0).to(self.device)

        if self.task == 'depth':
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(normal, size=(H, W), mode='bicubic', align_corners=False)
            normal = normal.cpu().numpy()
            return normal


# from munch import DefaultMunch
from midas.model_loader import default_models, load_model

depth_config={
    "input_path": None,
    "output_path": None,
    "model_weights": "pretrained/midas/dpt_beit_large_512.pt",
    "model_type": "dpt_beit_large_512",
    "side": False,
    "optimize": False,
    "height": None,
    "square": False,
    "device":0,
    "grayscale": False
}


class DepthEstimator:
    def __init__(self,**kwargs):
        # update coming args
        for key, value in kwargs.items():
            depth_config[key]=value
            
        # self.config=DefaultMunch.fromDict(depth_config)
        self.config = edict(depth_config) 
        
        # select device
        self.device = torch.device(self.config.device)
        model, transform, net_w, net_h = load_model(f"cuda:{self.config.device}", self.config.model_weights, self.config.model_type, 
                                                    self.config.optimize, self.config.height, self.config.square)
        self.model, self.transform, self.net_w, self.net_h=model, transform, net_w, net_h
        self.first_execution = True
        
    @torch.no_grad()
    def process(self,image,target_size):
        sample = torch.from_numpy(image).to(self.device).unsqueeze(0)


        if self.first_execution:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            self.first_execution = False

        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return prediction
    
    @torch.no_grad()
    def get_monocular_depth(self,rgb, output_path=None):
        original_image_rgb=rgb
        image = self.transform({"image": original_image_rgb})["image"]
        
        prediction = self.process(image, original_image_rgb.shape[1::-1])
        return prediction
    


def process_single_image(image_path, depth_estimator, normal_estimator=None):
    out_dir = os.path.dirname(image_path)
    rgba_path = os.path.join(out_dir, 'rgba.png')
    depth_path = os.path.join(out_dir, 'depth.png')
    # out_normal = os.path.join(out_dir, 'normal.png')

    if os.path.exists(rgba_path):
        print(f'[INFO] loading rgba image {rgba_path}...')
        rgba = cv2.cvtColor(cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    else:
        print(f'[INFO] loading image {image_path}...')
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f'[INFO] background removal...')
        rgba = BackgroundRemoval()(image)  # [H, W, 4]
        cv2.imwrite(rgba_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        # rgba = get_rgba(image)  # [H, W, 4]
        # cv2.imwrite(rgba_path.replace('rgba', 'rgba2'), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    # Predict depth using Midas
    mask = rgba[..., -1] > 0
    depth = depth_estimator.get_monocular_depth(image/255)
    depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
    depth[~mask] = 0
    depth = (depth * 255).astype(np.uint8)

    # print(f'[INFO] normal estimation...')
    # normal = normal_estimator(image)[0]
    # normal = (normal.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    # normal[~mask] = 0
    
    cv2.imwrite(depth_path, depth)
    # cv2.imwrite(out_normal, cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
    if not os.path.exists(rgba_path):
        cv2.imwrite(rgba_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

if __name__ == '__main__':
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str, nargs='*', help="path to image (png, jpeg, etc.)")
    parser.add_argument('--folder', default=None, type=str, help="path to image (png, jpeg, etc.)")
    opt = parser.parse_args()

    depth_estimator = DepthEstimator()
    # normal_estimator = DPT(task='normal')
    
    paths = opt.path if opt.path is not None else glob.glob(os.path.join(opt.folder, '*/rgba.png')) 
    for path in paths:
        process_single_image(path, depth_estimator, 
                            #  normal_estimator
                             )