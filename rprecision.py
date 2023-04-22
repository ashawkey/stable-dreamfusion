from sentence_transformers import SentenceTransformer, util
from PIL import Image
import argparse
import sys
# python r_precision.py --text "a snake is flying in the sky" --workspace snake_HQ --latest ep0100 --mode rgb
#python r_precision.py --text "a snake is flying in the sky" --workspace snake_HQ --latest ep0100 --mode depth

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="", type=str, help="text prompt")
    parser.add_argument('--workspace', default="trial", type=str, help="text prompt")
    parser.add_argument('--latest', default='ep0001', type=str, help="which epoch result you want to use for image path")
    parser.add_argument('--mode', default='rgb', type=str, help="mode of result, color(rgb) or textureless()")

    opt = parser.parse_args()

    #Load CLIP model
    model = SentenceTransformer('clip-ViT-B-32')

    #Encode an image:
    img_emb = model.encode(Image.open(f'results/{opt.workspace}/validation/df_{opt.latest}_0005_{opt.mode}.png'))

    #Encode text descriptions
    text_emb = model.encode([f'{opt.text}'])

    #Compute cosine similarities
    cos_scores = util.cos_sim(img_emb, text_emb)
    print("The final CLIP R-Precision is:", int(cos_scores))
