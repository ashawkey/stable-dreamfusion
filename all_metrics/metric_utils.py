# * evaluate use laion/CLIP-ViT-H-14-laion2B-s32B-b79K
# best open source clip so far: laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
# code adapted from NeuralLift-360

import torch
import torch.nn as nn
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
# import clip
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPProcessor
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image
# import torchvision.transforms as transforms
import glob
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
from os.path import join as osp
import argparse
import pandas as pd

class CLIP(nn.Module):

    def __init__(self,
                 device,
                 clip_name='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
                 size=224):  #'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'):
        super().__init__()
        self.size = size
        self.device = f"cuda:{device}"

        clip_name = clip_name

        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-base-patch32')

        self.normalize = transforms.Normalize(
            mean=self.feature_extractor.image_mean,
            std=self.feature_extractor.image_std)

        self.resize = transforms.Resize(224)
        self.to_tensor = transforms.ToTensor()

        # image augmentation
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])

    # * recommend to use this function for evaluation
    @torch.no_grad()
    def score_gt(self, ref_img_path, novel_views):
        # assert len(novel_views) == 100
        clip_scores = []
        for novel in novel_views:
            clip_scores.append(self.score_from_path(ref_img_path, [novel]))
        return np.mean(clip_scores)

    # * recommend to use this function for evaluation
    # def score_gt(self, ref_paths, novel_paths):
    #     clip_scores = []
    #     for img1_path, img2_path in zip(ref_paths, novel_paths):
    #         clip_scores.append(self.score_from_path(img1_path, img2_path))

    #     return np.mean(clip_scores)

    def similarity(self, image1_features: torch.Tensor,
                   image2_features: torch.Tensor) -> float:
        with torch.no_grad(), torch.cuda.amp.autocast():
            y = image1_features.T.view(image1_features.T.shape[1],
                                       image1_features.T.shape[0])
            similarity = torch.matmul(y, image2_features.T)
            # print(similarity)
            return similarity[0][0].item()

    def get_img_embeds(self, img):
        if img.shape[0] == 4:
            img = img[:3, :, :]

        img = self.aug(img).to(self.device)
        img = img.unsqueeze(0)  # b,c,h,w

        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).numpy())
        # plt.show()
        # print(img)

        image_z = self.clip_model.get_image_features(img)
        image_z = image_z / image_z.norm(dim=-1,
                                         keepdim=True)  # normalize features
        return image_z

    def score_from_feature(self, img1, img2):
        img1_feature, img2_feature = self.get_img_embeds(
            img1), self.get_img_embeds(img2)
        # for debug
        return self.similarity(img1_feature, img2_feature)

    def read_img_list(self, img_list):
        size = self.size
        images = []
        # white_background = np.ones((size, size, 3), dtype=np.uint8) * 255

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # print(img_path)
            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)  # Convert BGRA to BGR
                img[np.where(alpha == 0)] = [
                    255, 255, 255
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

            # plt.imshow(img)
            # plt.show()

            images.append(img)

        images = np.stack(images, axis=0)
        # images[np.where(images == 0)] = 255  # Set black pixels to white
        # images = np.where(images == 0, white_background, images)  # Set transparent pixels to white
        # images = images.astype(np.float32)

        return images

    def score_from_path(self, img1_path, img2_path):
        img1, img2 = self.read_img_list(img1_path), self.read_img_list(img2_path)
        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)
        # plt.imshow(img1)
        # plt.show()
        # plt.imshow(img2)
        # plt.show()

        img1, img2 = self.to_tensor(img1), self.to_tensor(img2)
        # print("img1 to tensor ",img1)
        return self.score_from_feature(img1, img2)


def numpy_to_torch(images):
    images = images * 2.0 - 1.0
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
    return images.cuda()


class LPIPSMeter:

    def __init__(self,
                 net='alex',
                 device=None,
                 size=224):  # or we can use 'alex', 'vgg' as network
        self.size = size
        self.net = net
        self.results = []
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def measure(self):
        return np.mean(self.results)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

    def read_img_list(self, img_list):
        size = self.size
        images = []
        white_background = np.ones((size, size, 3), dtype=np.uint8) * 255

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img[np.where(alpha == 0)] = [
                    255, 255, 255
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            images.append(img)

        images = np.stack(images, axis=0)
        # images[np.where(images == 0)] = 255  # Set black pixels to white
        # images = np.where(images == 0, white_background, images)  # Set transparent pixels to white
        images = images.astype(np.float32) / 255.0

        return images

    # * recommend to use this function for evaluation
    @torch.no_grad()
    def score_gt(self, ref_paths, novel_paths):
        self.results = []
        for path0, path1 in zip(ref_paths, novel_paths):
            # Load images
            # img0 = lpips.im2tensor(lpips.load_image(path0)).cuda() # RGB image from [-1,1]
            # img1 = lpips.im2tensor(lpips.load_image(path1)).cuda()
            img0, img1 = self.read_img_list([path0]), self.read_img_list(
                [path1])
            img0, img1 = numpy_to_torch(img0), numpy_to_torch(img1)
            # print(img0.shape,img1.shape)
            img0 = F.interpolate(img0,
                                    size=(self.size, self.size),
                                    mode='area')
            img1 = F.interpolate(img1,
                                    size=(self.size, self.size),
                                    mode='area')

            # for debug vis
            # plt.imshow(img0.cpu().squeeze(0).permute(1, 2, 0).numpy())
            # plt.show()
            # plt.imshow(img1.cpu().squeeze(0).permute(1, 2, 0).numpy())
            # plt.show()
            # equivalent to cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA

            # print(img0.shape,img1.shape)

            self.results.append(self.fn.forward(img0, img1).cpu().numpy())

        return self.measure()


class PSNRMeter:

    def __init__(self, size=800):
        self.results = []
        self.size = size

    def read_img_list(self, img_list):
        size = self.size
        images = []
        white_background = np.ones((size, size, 3), dtype=np.uint8) * 255
        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                img = cv2.cvtColor(img,
                                   cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img[np.where(alpha == 0)] = [
                    255, 255, 255
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            images.append(img)

        images = np.stack(images, axis=0)
        # images[np.where(images == 0)] = 255  # Set black pixels to white
        # images = np.where(images == 0, white_background, images)  # Set transparent pixels to white
        images = images.astype(np.float32) / 255.0
        # print(images.shape)
        return images

    def update(self, preds, truths):
        # print(preds.shape)

        psnr_values = []
        # For each pair of images in the batches
        for img1, img2 in zip(preds, truths):
            # Compute the PSNR and add it to the list
            # print(img1.shape,img2.shape)

            # for debug
            # plt.imshow(img1)
            # plt.show()
            # plt.imshow(img2)
            # plt.show()

            psnr = compare_psnr(
                img1, img2,
                data_range=1.0)  # assuming your images are scaled to [0,1]
            # print(f"temp psnr {psnr}")
            psnr_values.append(psnr)

        # Convert the list of PSNR values to a numpy array
        self.results = psnr_values

    def measure(self):
        return np.mean(self.results)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

    # * recommend to use this function for evaluation
    def score_gt(self, ref_paths, novel_paths):
        self.results = []
        # [B, N, 3] or [B, H, W, 3], range[0, 1]
        preds = self.read_img_list(ref_paths)
        truths = self.read_img_list(novel_paths)
        self.update(preds, truths)
        return self.measure()

all_inputs = 'data'
nerf_dataset = os.listdir(osp(all_inputs, 'nerf4'))
realfusion_dataset = os.listdir(osp(all_inputs, 'realfusion15'))
meta_examples = {
   'nerf4': nerf_dataset, 
   'realfusion15': realfusion_dataset, 
}
all_datasets = meta_examples.keys()

# organization 1
def deprecated_score_from_method_for_dataset(my_scorer,
                                  method,
                                  dataset,
                                  input,
                                  output,
                                  score_type='clip', 
                                  ):  # psnr, lpips
    # print("\n\n\n")
    # print(f"______{method}___{dataset}___{score_type}_________")
    scores = {}
    final_res = 0
    examples = meta_examples[dataset]
    for i in range(len(examples)):
        
        # compare entire folder for clip
        if score_type == 'clip':
            novel_view = osp(pred_path, examples[i], 'colors')
        # compare first image for other metrics
        else:
            if method == '3d_fuse': method = '3d_fuse_0'
            novel_view = list(
                glob.glob(
                    osp(pred_path, examples[i], 'colors',
                        'step_0000*')))[0]

        score_i = my_scorer.score_gt(
            [], [novel_view])
        scores[examples[i]] = score_i
        final_res += score_i
    # print(scores, " Avg : ", final_res / len(examples))
    # print("``````````````````````")
    return scores

# results organization 2
def score_from_method_for_dataset(my_scorer,
                                  input_path,
                                  pred_path,
                                  score_type='clip', 
                                  rgb_name='lambertian', 
                                  result_folder='results/images', 
                                  first_str='*0000*'
                                  ):  # psnr, lpips
    scores = {}
    final_res = 0
    examples = os.listdir(input_path)
    for i in range(len(examples)):
        # ref path
        ref_path = osp(input_path, examples[i], 'rgba.png') 
        # compare entire folder for clip
        if score_type == 'clip':
            novel_view = glob.glob(osp(pred_path,'*'+examples[i]+'*', result_folder, f'*{rgb_name}*'))
            print(f'[INOF] {score_type} loss for example {examples[i]} between 1 GT and {len(novel_view)} predictions')
        # compare first image for other metrics
        else:
            novel_view = glob.glob(osp(pred_path, '*'+examples[i]+'*/', result_folder, f'{first_str}{rgb_name}*'))
            print(f'[INOF] {score_type} loss for example {examples[i]} between {ref_path} and {novel_view}')
        # breakpoint()
        score_i = my_scorer.score_gt([ref_path], novel_view)
        scores[examples[i]] = score_i
        final_res += score_i
    avg_score = final_res / len(examples)
    scores['average'] = avg_score
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to accept three string arguments")
    parser.add_argument("--input_path",
                        default=all_inputs,
                        help="Specify the input path")
    parser.add_argument("--pred_pattern",
                        default="out/magic123*", 
                        help="Specify the pattern of predition paths")
    parser.add_argument("--results_folder",
                        default="results/images", 
                        help="where are the results under each pred_path")
    parser.add_argument("--rgb_name",
                        default="lambertian", 
                        help="the postfix of the image")
    parser.add_argument("--first_str",
                        default="*0000*", 
                        help="the str to indicate the first view")
    parser.add_argument("--datasets",
                        default=all_datasets,
                        nargs='*',
                        help="Specify the output path")
    parser.add_argument("--device",
                        type=int,
                        default=0,
                        help="Specify the GPU device to be used")
    parser.add_argument("--save_dir", type=str, default='all_metrics/results')
    args = parser.parse_args()

    clip_scorer = CLIP(args.device)
    lpips_scorer = LPIPSMeter()
    psnr_scorer = PSNRMeter()

    os.makedirs(args.save_dir, exist_ok=True)

    for dataset in args.datasets:
        input_path = osp(args.input_path, dataset)
        
        # assume the pred_path is organized as: pred_path/methods/dataset
        pred_pattern = osp(args.pred_pattern, dataset)
        pred_paths = glob.glob(pred_pattern)
        print(f"[INFO] Following the pattern {pred_pattern}, find {len(pred_paths)} pred_paths: \n", pred_paths)
        if len(pred_paths) == 0:
            raise IOError
        for pred_path in pred_paths:
            if not os.path.exists(pred_path):
                print(f'[WARN] prediction does not exit for {pred_path}')
            else:
                print(f'[INFO] evaluate {pred_path}')
            results_dict = {}
            results_dict['clip'] = score_from_method_for_dataset(
                clip_scorer, input_path, pred_path, 'clip', 
                result_folder=args.results_folder, rgb_name=args.rgb_name, first_str=args.first_str)
            
            results_dict['psnr'] = score_from_method_for_dataset(
                psnr_scorer, input_path, pred_path,  'psnr', 
                result_folder=args.results_folder, rgb_name=args.rgb_name, first_str=args.first_str)
            
            results_dict['lpips'] = score_from_method_for_dataset(
                lpips_scorer, input_path, pred_path,  'lpips', 
                result_folder=args.results_folder, rgb_name=args.rgb_name, first_str=args.first_str)
            
            df = pd.DataFrame(results_dict)
            method = pred_path.split('/')[-2]
            print(osp(pred_path, args.results_folder))
            results_str = '_'.join(args.results_folder.split('/'))
            print(method+'-'+results_str)
            print(df)
            df.to_csv(f"{args.save_dir}/{method}-{results_str}-{dataset}.csv")