import os
import glob
import tqdm
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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
    
class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


class DPT():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

    @torch.no_grad()
    def __call__(self, image):

        H, W = image.shape[:2]
        image = Image.fromarray(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        outputs = self.model(pixel_values)
        depth = outputs.predicted_depth
        depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
        depth = depth.squeeze(1).cpu().numpy()

        return depth


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    opt = parser.parse_args()
    
    out_dir = os.path.dirname(opt.path)
    out_rgba = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_rgba.png')
    out_depth = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_depth.png')
    out_caption = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_caption.txt')

    # load image
    print(f'[INFO] loading image...')
    image = cv2.imread(opt.path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # predict depth (with bg as it's more stable)
    print(f'[INFO] depth estimation...')
    dpt_model = DPT()
    depth = dpt_model(image)[0]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth = (depth * 255).astype(np.uint8)
    cv2.imwrite(out_depth, depth)

    # predict caption (it's too slow... use your brain instead)
    # print(f'[INFO] captioning...')
    # blip2 = BLIP2()
    # caption = blip2(image)
    # with open(out_caption, 'w') as f:
    #     f.write(caption)

    # carve background
    print(f'[INFO] background removal...')
    image = BackgroundRemoval()(image) # [H, W, 4]
    
    cv2.imwrite(out_rgba, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))


    

