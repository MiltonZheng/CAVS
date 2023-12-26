import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from clip import tokenize, load
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.transforms import Resize
from PIL import Image



class myDataset(Dataset):
    def __init__(self):
        dataset = "/root/autodl-tmp/datasets/mirflickr"
        data_type = "test"
        self.file_img = open(dataset + f'/{data_type}_img.npy', 'rb')
        self.file_text = open(dataset + f'/{data_type}_text.npy', 'rb')
        self.file_label = open(dataset + f'/{data_type}_label.npy', 'rb')
        self.images = np.load(self.file_img)
        self.texts = np.load(self.file_text)
        self.labels = np.load(self.file_label)
        self.file_img.close()
        self.file_text.close()
        self.file_label.close()
        
    def __getitem__(self, index):
        image = self.images[index, :]
        text = self.texts[index, :]
        # to one-hot
        label = self.labels[index, :]
        return image, text, label

    def __len__(self):
        return len(self.images)
        

def extract_features_w_clip(images):
    clip, transforms = load('ViT-B/32')
    features = []
    # call the DataLoader to read data iteratively
    for id, image in enumerate(tqdm(images, desc="extracting features")):
        image = np.transpose(image, (2, 1, 0))
        image = Image.fromarray(image)
        image = transforms(image).unsqueeze(0).to(torch.device("cuda"))
        feature = clip.encode_image(image)
        feature = torch.Tensor(feature).detach().cpu().numpy()
        features.extend(feature)

    # store the features as hy file
    features = np.asarray(features)
    print("feature size:{}".format(features.shape))
    features_h5 = h5py.File(os.path.join("/root/autodl-tmp/datasets/mirflickr", 'clip_features.hy'), 'w')
    features_h5.create_dataset("features", data=features)
    features_h5.close()
    return features

def extract_text_feature(text):
    token = tokenize(text)
    clip, transforms = load('ViT-B/32')
    text_feature = clip.encode_text(token)
    return np.asarray(text_feature)
    

if __name__ == "__main__":
    file_img = open("/root/autodl-tmp/datasets/mirflickr" + '/test_img.npy', 'rb')
    images = np.load(file_img)
    extract_features_w_clip(images)
    pass