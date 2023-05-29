import torch
import torch.nn as nn

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import clip

class CLIP(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()

        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False)

        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def get_text_embeds(self, prompt, **kwargs):

        text = clip.tokenize(prompt).to(self.device)
        text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        return text_z

    def get_img_embeds(self, image, **kwargs):

        image_z = self.clip_model.encode_image(self.aug(image))
        image_z = image_z / image_z.norm(dim=-1, keepdim=True)

        return image_z

    
    def train_step(self, clip_z, pred_rgb, grad_scale=10, **kwargs):
        """
            Args:
                grad_scale: scalar or 1-tensor of size [B], i.e. 1 grad_scale per batch item. 
        """
        # TODO: resize the image from NeRF-rendered resolution (e.g. 128x128) to what CLIP expects (512x512), to prevent Pytorch warning about `antialias=None`.
        image_z = self.clip_model.encode_image(self.aug(pred_rgb))
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        loss = 0
        if 'image' in clip_z:
            loss -= ((image_z * clip_z['image']).sum(-1) * grad_scale).mean()
        
        if 'text' in clip_z:
            loss -= ((image_z * clip_z['text']).sum(-1) * grad_scale).mean()

        return loss

