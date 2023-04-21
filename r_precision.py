from sentence_transformers import SentenceTransformer, util
from PIL import Image
import argparse
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="", type=str, help="text prompt")
    parser.add_argument('--workspace', default="trial", type=str, help="text prompt")
    parser.add_argument('--latest', default='ep0001', type=str, help="which epoch result you want to use for image path")
    parser.add_argument('--mode', default='rgb', type=str, help="mode of result, color(rgb) or textureless()")
    parser.add_argument('--clip', default="clip-ViT-B-32", type=str, help="CLIP model to encode the img and prompt")

    opt = parser.parse_args()

    #Load CLIP model
    model = SentenceTransformer(f'{opt.clip}')

    #Encode an image:
    img_emb = model.encode(Image.open(f'results/{opt.workspace}/validation/df_{opt.latest}_0005_{opt.mode}.png'))

    #Encode text descriptions
    text_emb = model.encode([f'{opt.text}'])

    #Compute cosine similarities
    cos_scores = util.cos_sim(img_emb, text_emb)
    print("The final CLIP R-Precision is:", cos_scores[0][0].cpu().numpy())

