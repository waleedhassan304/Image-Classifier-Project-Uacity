import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['vgg_type'] == "vgg16":  # Adjust based on the available model types
        model = torchvision.models.vgg16(pretrained=True)
    else:
        print("Model type not recognized.")
        return None

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    pil_image = Image.open(image_path)
    pil_image = pil_image.resize((256, 256))
    pil_image = pil_image.crop((16, 16, 240, 240))
    np_image = np.array(pil_image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)
    return tensor

def predict(image_path, model, topk, device, cat_to_name):
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image))

    ps, top_classes = ps.topk(topk, dim=1)
    idx_to_flower = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes.tolist()[0]]
    return ps.tolist()[0], predicted_flowers_list

def print_predictions(args):
    model = load_checkpoint(args.model_filepath)

    if model is None:
        print("Error loading the model. Check your model type.")
        return

    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    if args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'

    model = model.to(device)

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    print("Predictions:")
    for i in range(args.top_k):
        print("#{: <3} {: <25} Prob: {:.2f}%".format(i, top_classes[i], top_ps[i] * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='image_filepath', help="Path to the image file you want to classify")
    parser.add_argument(dest='model_filepath', help="Path to the checkpoint file, including the extension")

    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath',
                        help="Path to a json file that maps categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="Number of most likely classes to return, default is 5", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to use the GPU for inference", action='store_true')

    args = parser.parse_args()
    print_predictions(args)
