import os, subprocess
import sys
import torch
from PIL import Image
import open_clip
from clip_interrogator import Config, Interrogator
# import gradio as gr

caption_model_name = ""
clip_model_name = "ViT-B-32/open_clip_pytorch_model.bin" #@param ["ViT-B-16/openai", "ViT-B-32/openai"]
mode = 'classic'

config = Config()
config.clip_model_name = clip_model_name
config.caption_model_name = caption_model_name
ci = Interrogator(config)

def image_analysis(image):
    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_benign= ci.benign.rank(image_features, 5)
    top_system = ci.system.rank(image_features, 5)
    top_diagnoses = ci.diagnoses.rank(image_features, 5)

    benign_ranks = {benign: sim for benign, sim in zip(top_benign, ci.similarities(image_features, top_benign))}
    system_ranks = {system: sim for system, sim in zip(top_system, ci.similarities(image_features, top_system))}
    diagnoses_ranks = {diagnoses: sim for diagnoses, sim in zip(top_diagnoses, ci.similarities(image_features, top_diagnoses))}
    
    return benign_ranks, system_ranks, diagnoses_ranks

def image_to_prompt(image, mode):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)
    
if __name__ == "__main__":
    # get arguments from command line
    args = sys.argv[1:]
    image_dir = 'images'
    image_path = os.path.join(image_dir, args[0])
    print(image_path)
    # convert path to PIL image
    image = Image.open(image_path)
    
    benign_ranks, system_ranks, diagnoses_ranks = image_analysis(image)
    
    print(benign_ranks)
    print(system_ranks)
    print(diagnoses_ranks)
    
    print(image_to_prompt(image, mode))
