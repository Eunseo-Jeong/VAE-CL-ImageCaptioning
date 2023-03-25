import os
import json
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image 
from torchvision.io import read_image
from collections import Counter  # list의 단어 count해주는 class


class newDataset(Dataset):
    def __init__(self, path, mode):
        self.mode = mode
        
        if self.mode  == "train":
            annotation_path = path['annotation']
            self.image_path = path['image']
        elif self.mode  == "val":
            annotation_path = path['annotation']
            self.image_path = path['image']
        # elif self.mode == 'test':
            
       
        with open(annotation_path, "r") as file:
            self.annotation = json.load(file)
        
        
        # 먼저, 변환하기 전 이미지 데이터셋을 로드 하기 위하여 transforms.ToTensor() 만 적용합니다.
        self.transform = transforms.Compose([
            transforms.Resize([224, 224])
            # transforms.ToTensor()
        ])
        
        if self.mode == 'train':
            self.annotation = self.annotation["annotations"][:50000] # images, licenses, annotations
        elif self.mode == 'val':
            self.annotation = self.annotation["annotations"][:10000]

        # # answer_list
        # self.answer_list = []

        # for i in range(len(self.annotation)):
        #     answer = self.annotation[i]['multiple_choice_answer']
        #     self.answer_list.append(answer)
        
        # self.label_answer = dict(Counter(self.answer_list)) # 2522
        # self.label_answer['unk'] = 0

        # # 2522
        # self.label_answer_list = self.label_answer.keys()
        # # print(len(self.label_answer_list))
        # self.label_answer_id2idx = {j: i for i, j in enumerate(self.label_answer_list)}
        # # print(len(self.label_answer_id2idx))
        # self.label_answer_idx2id = {i: j for i, j in enumerate(self.label_answer_list)}
        # # print(len(self.label_answer_idx2id))

            
    def __len__(self):
        return len(self.annotation) # 60000


    def __getitem__(self, idx):
        
        image_id = self.annotation[idx]['image_id']
        id = self.annotation[idx]['id']
        caption = self.annotation[idx]['caption']
        
        
        # image
        if self.mode == 'train':
            img_str = str(image_id).zfill(12)
            img_str = "COCO_train2014_" + img_str + ".jpg"
            img_path = os.path.join(self.image_path, img_str)
        elif self.mode == 'val':
            img_str = str(image_id).zfill(12)
            img_str = "COCO_val2014_" + img_str + ".jpg"
            img_path = os.path.join(self.image_path, img_str)
        # elif self.mode == 'test':
        #     img_str = str(image_id).zfill(12)
        #     img_str = "COCO_test2014_" + img_str + ".jpg"
        #     img_path = os.path.join(self.image_path, img_str)


        # # image
        # if self.mode == 'train':
        #     img_str = str(image_id).zfill(12)
        #     img_str = "COCO_train2014_" + img_str + ".jpg" # abstract_v002_train2015
        #     img_path = os.path.join(self.image_path, img_str)
        # elif self.mode == 'val':
        #     img_str = str(image_id).zfill(12)
        #     img_str = "COCO_val2014_" + img_str + ".jpg"
        #     img_path = os.path.join(self.image_path, img_str)
        
        
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        image = self.transform(image) # 224, 224

        return img_path, image, image_id, id, caption


    def collate_fn(self, batch):
        img_path, image, image_id, id, caption = zip(*batch)
        return img_path, image, image_id, id, caption