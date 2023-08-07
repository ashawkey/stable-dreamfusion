import torch
import torch.nn as nn

import torchvision.transforms as T
import torchvision.transforms.functional as TF

# import clip
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPProcessor
from torchvision import transforms

import torch.nn.functional as F


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    # print(x.shape, y.shape)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class CLIP(nn.Module):
    def __init__(self, device, 
                #  clip_name = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
                 clip_name = 'openai/clip-vit-large-patch14'
                 ):
        super().__init__()

        self.device = device

        clip_name = clip_name

        # self.feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).cuda()
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        # self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
 
        # self.normalize = transforms.Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)

        # self.resize = transforms.Resize(224)

        #  # image augmentation
        # self.aug = T.Compose([
        #     T.Resize((224, 224)),
        #     T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])

    
    def get_text_embeds(self, prompt, neg_prompt=None, dir=None):

        clip_text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        text_z = self.clip_model.get_text_features(clip_text_input)
        # text = clip.tokenize(prompt).to(self.device)
        # text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        return text_z

    def set_epoch(self, epoch):
        pass

    def get_img_embeds(self, img):
        img = self.aug(img)
        image_z = self.clip_model.get_image_features(img)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features
        return image_z

    
    # def train_step(self, text_z, pred_rgb, image_ref_clip, **kwargs):

    #     pred_rgb = self.resize(pred_rgb)
    #     pred_rgb = self.normalize(pred_rgb)

    #     image_z = self.clip_model.get_image_features(pred_rgb)
    #     image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

    #     # print(image_z.shape, text_z.shape)
    #     loss = spherical_dist_loss(image_z, image_ref_clip)

    #     # loss = - (image_z * text_z).sum(-1).mean()

    #     return loss
    
    def train_step(self, text_z, pred_rgb):

        pred_rgb = self.aug(pred_rgb)

        image_z = self.clip_model.encode_image(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        loss = - (image_z * text_z).sum(-1).mean()
        # loss = spherical_dist_loss(image_z, text_z)
        return loss
    
    def text_loss(self, text_z, pred_rgb):

        pred_rgb = self.resize(pred_rgb)
        pred_rgb = self.normalize(pred_rgb)

        image_z = self.clip_model.get_image_features(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        # print(image_z.shape, text_z.shape)
        loss = spherical_dist_loss(image_z, text_z)

        # loss = - (image_z * text_z).sum(-1).mean()

        return loss
    
    def img_loss(self, img_ref_z, pred_rgb):
        # pred_rgb = self.aug(pred_rgb)
        pred_rgb = self.resize(pred_rgb)
        pred_rgb = self.normalize(pred_rgb)

        image_z = self.clip_model.get_image_features(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        # loss = - (image_z * img_ref_z).sum(-1).mean()
        loss = spherical_dist_loss(image_z, img_ref_z)

        return loss
