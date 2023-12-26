import os
import cv2
import h5py
import json
import torch
import scipy
import requests
import numpy as np
import torch.nn.functional as F
from base64 import b64encode
from clip import tokenize, load


def extract_text_feature(text):
    token = tokenize(text, truncate=True).to(torch.device("cuda"))
    clip, transforms = load('ViT-B/32')
    text_feature = clip.encode_text(token)
    text_feature = text_feature.detach().cpu().numpy()
    return text_feature

def get_description(img_path):
    with open(img_path, 'rb') as f:  # 此处不能使用encoding='utf-8'， 否则报错
        image = b64encode(f.read())  # b64encode是编码
    image = str(image, encoding="utf-8")

    req = {"image" : image,
           "text" : "sum up this photo",
           "history" : []
           }

    response = requests.post("http://127.0.0.1:8080", json=req)
    j = json.loads(response.text)
    return j["result"]


def cal_dist(qu, re):
    qu_2 = np.dot(qu, re.transpose())
    return qu_2


if __name__ == "__main__":
    data_path = "/root/autodl-tmp/datasets/mirflickr"
    img_fea = h5py.File(os.path.join(data_path, "clip_features.hy"), "r")
    img_fea = img_fea["features"][()]
    img_file = open(os.path.join(data_path, 'test_img.npy'), 'rb')
    img = np.load(img_file)
    label = scipy.io.loadmat(os.path.join(data_path, 'mirflickr25k-lall.mat'))["LAll"]
    mAP, mAP_img = 0.0, 0.0
    p, p_img = 0.0, 0.0
    query_num = 1
    for index in [4]:
        print(index)
        # get description
        to_save = np.transpose(img[index], (2, 1, 0))
        to_save = cv2.cvtColor(to_save, cv2.COLOR_BGR2RGB)
        img_path = os.path.join(data_path, "test.jpg")
        cv2.imwrite(img_path, to_save)
        desc = get_description(img_path)
        print(desc)
        txt_fea = extract_text_feature(desc)
    
        # imgs that share the same tag with the query image
        ground = (np.dot(label[index, :], label.transpose()) > 0).astype(np.float32)
        
        # calculate and sort distance
        dists = cal_dist(txt_fea, img_fea).reshape(-1)
        dists_img = cal_dist(img_fea[index], img_fea).reshape(-1)
        idx = np.argsort(dists)
        idx_img = np.argsort(dists_img)
        ground_txt = ground[idx]
        ground_img = ground[idx_img]
        
        total = np.sum(ground)
        print(total)
        k = 10
        top_k = ground_txt[:k]
        top_k_img = ground_img[:k]
        sum = np.sum(top_k)
        sum_img = np.sum(top_k_img)
    
        if sum == 0 or sum_img == 0:
            topkmap = 0.0
            precision = 0.0
        else:
            count = np.linspace(1, sum, int(sum))
            tindex = np.asarray(np.where(top_k == 1)) + 1.0
            topkmap = np.mean(count / (tindex))
            precision = sum / k
        p += precision
        mAP += topkmap
        
        if sum_img == 0:     
            topkmap_img = 0.0
            precision_img = 0.0
        else:
            count_img = np.linspace(1, sum_img, int(sum_img))
            tindex_img = np.asarray(np.where(top_k_img == 1)) + 1.0
            topkmap_img = np.mean(count_img / (tindex_img))
            precision_img = sum_img / k
        p_img += precision_img
        mAP_img += topkmap_img
        
        print(f"mAP = {topkmap}, precision = {precision}")
            
    print(f"k = {10}, ")
    print(f"precision = {p/query_num}, mAP@{k} = {mAP/query_num}")
    print(f"precision = {p_img/query_num}, mAP@{k} = {mAP_img/query_num}")
    pass
    
    
    