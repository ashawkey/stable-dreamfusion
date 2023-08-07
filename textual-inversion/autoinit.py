"""
It takes about 2 minutes to compute and save embeddings for all noun tokens in the CLIP tokenizer vocabulary. Examples:

python autoinit.py save_embeddings
python autoinit.py get_initialization /path/to/bird.jpg
 
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

torch.set_grad_enabled(False)

DEFAULT_EMB_FILE = 'clip-vit-large-patch14-text-embeddings.pth'


def get_model():
    model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    processor: CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


def save_embeddings(file_name: str = DEFAULT_EMB_FILE, device: str = 'cuda'):
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        print('Please install google fire with `pip install fire`')
        sys.exit()

    # # The first time you run this code you will have to run this
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # All English nouns
    english_nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
    print(f'Found {len(english_nouns)} English nouns')

    # Get model
    model, processor = get_model()
    model.to(device)

    # Get all tokens in CLIP tokenizer that are nouns
    all_noun_ids = []
    all_token_ids = sorted(processor.tokenizer.vocab.values())
    for token_id in tqdm(all_token_ids):
        token_str = processor.tokenizer.convert_ids_to_tokens(token_id)
        if token_str.replace('</w>', '') in english_nouns and token_str.endswith('</w>'):
            all_noun_ids.append(token_id)
    print(f'Found {len(all_noun_ids)} English nouns in the CLIP tokenizer')

    # Get all embeddings
    all_text_emb = []
    all_text_str = []
    for token_id in tqdm(all_noun_ids):
        text_ids = [49406, 550, 2867, 539, 320, token_id, 49407]  # "<bos> an image of a _ <eos>"
        text_str = processor.tokenizer.decode(text_ids, skip_special_tokens=True)
        inputs = processor(text=text_str, return_tensors="pt", padding=True)
        text_emb = model.get_text_features(**inputs.to(device))
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        all_text_emb.append(text_emb.detach().cpu())
        all_text_str.append(text_str)
    all_text_emb = torch.cat(all_text_emb)

    # Save
    torch.save({
        'idx': all_noun_ids,
        'emb': all_text_emb,
    }, file_name)
    print(f'Saved embeddings to {file_name}')

# %%

def get_initialization(image_file: str, text_emb_file: str = DEFAULT_EMB_FILE, device: str = 'cuda', 
                       save: bool = False, save_dir: Optional[str] = None):

    # Load text embeddings
    text_emb = torch.load(text_emb_file)
    all_noun_ids = text_emb['idx']
    all_noun_emb = text_emb['emb']

    # Get model
    model, processor = get_model()
    model.to(device)

    # Load and process
    image = Image.open(image_file)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_emb = model.get_image_features(**inputs.to(device))
    image_emb = F.normalize(image_emb, p=2, dim=-1)

    # Get similarities
    sim = all_noun_emb.to(device) @ image_emb.to(device).squeeze()  # (V, )
    sim = F.softmax(sim, dim=-1)  # (V, )
    topk_texts = sim.topk(k=5, largest=True, sorted=True)
    topk_indices = [all_noun_ids[idx] for idx in topk_texts.indices.cpu()]

    # Print topk
    topk_tokens = processor.tokenizer.convert_ids_to_tokens(topk_indices)
    top_token = topk_tokens[0].replace('</w>', '')
    print('Top tokens:')
    print(topk_tokens)
    if save:
        save_dir = Path(image_file).parent if save_dir is None else Path(save_dir)
        text_file = save_dir / 'token_autoinit.txt'
        text_file.write_text(top_token)

if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        print('Please install google fire with `pip install fire`')
        sys.exit()
    fire.Fire(dict(get_initialization=get_initialization, save_embeddings=save_embeddings))