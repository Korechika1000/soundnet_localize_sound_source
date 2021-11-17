from network import *
from matplotlib import pyplot as plt
import seaborn as sns
import os,glob,json
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional
from PIL import Image
from random import randint
from tqdm import tqdm
import time
import random
from random import choice
import math
import pdb
import scipy
import scipy.io as sio
import cv2


def audio_loader(sample, neg_sample):
    # Get positive audio
    video_path = sample.replace('\n', '')
    audio_path = video_path + '.mat'
    pos_sound_file = sio.loadmat(audio_path)
    #modify
    pos_sound = (pos_sound_file['a_feature'])
    pos_sound = np.asarray(pos_sound) # CHECK THE SIZE!
    pos_sound_tensor = torch.from_numpy(pos_sound).squeeze().float()
    # Get negative audio
    neg_video_path = neg_sample.replace('\n','')
    neg_audio_path = neg_video_path+'.mat'
    neg_sound_file = sio.loadmat(neg_audio_path)
    # modify
    neg_sound = (neg_sound_file['a_feature'])
    neg_sound = np.asarray(neg_sound)  # CHECK THE SIZE!
    neg_sound_tensor = torch.from_numpy(neg_sound).squeeze().float()
    return pos_sound_tensor, neg_sound_tensor

def image_loader(sample):
    video_path = sample.replace('\n','')
    all_frames = video_path+'.mp4'
    cap_file = cv2.VideoCapture(all_frames)
    frame_count = int(cap_file.get(cv2.CAP_PROP_FRAME_COUNT))
    first_image = []
    for _ in range(0, frame_count):
        _, frame = cap_file.read()
        # BGR -> RGB
        rgb_cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a PIL image object from a NumPy array
        pil_image = Image.fromarray(rgb_cv2_image)
        first_image.append(pil_image)
    return first_image


def show_heatmap_video(sample_path, result, cnt):
    print('Video export: {}'.format(cnt))
    cap_file = cv2.VideoCapture(sample_path)
    width = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_file.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap_file.get(cv2.CAP_PROP_FRAME_COUNT))
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('out_'+ str(cnt) +'.mp4', fmt, fps, (width, height))
    for jo in tqdm(range(0, frame_count)):
        ret, frame = cap_file.read()
        if ret == True:
            map1 = result[jo]
            remap1 = cv2.resize(map1,(width,height))
            heatmap = np.uint8(remap1)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            frame_map = frame*0.5+heatmap*0.5
            frame_map = frame_map.astype(np.uint8)
            writer.write(frame_map)
        else:
            break
    cap_file.release()
    writer.release()
    return ret
        

class Sound_Localization_Dataset(Dataset):
    def __init__(self, dataset_file):
        ds = open(dataset_file)
        lines = ds.readlines()
        #print(lines)
        self.data = lines
        self.preprocess = transforms.Compose([transforms.Resize((320,320)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __getitem__(self,index):
        datum = self.data[index]
        # Get negative index and negative sample
        neg_index = choice([r for r in range(0,len(self.data)) if r not in [index]])
        neg_datum = self.data[neg_index]
        # Get video frames
        first_frame = image_loader(datum)
        first_frame_t = []
        for m in range(0, len(first_frame)):
            f = self.preprocess(first_frame[m]).float()
            first_frame_t.append(f)
        pos_audio_t,neg_audio_t = audio_loader(datum, neg_datum)
        return first_frame_t, pos_audio_t, neg_audio_t, datum


    def __len__(self):
        return len(self.data)


def main():
    cnt = 0
    sns.set()
    # load model
    net = AVModel()
    net.load_state_dict(torch.load("sound_localization_latest.pth"))
    # load data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    val_dataset_file = os.path.abspath(os.path.join(BASE_DIR,'myvideo', "mytest_video.txt"))
    dataset_test = Sound_Localization_Dataset(val_dataset_file)
    dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle = False)
    #validation
    for i, mydata in enumerate(dataloader_test):
        result = []
        print('\nEval step:', i,mydata[3])
        for im in tqdm(mydata[0]):
            frame_t_val = im
            pos_audio_val = mydata[1]
            neg_audio_val = mydata[2]
            z_val, pos_audio_embedding_val, neg_audio_embedding_val, att_map_val = net.forward(frame_t_val, pos_audio_val, neg_audio_val)   
            att_map = att_map_val.view(20,20)
            b = att_map.detach().numpy()
            norm_image = cv2.normalize(b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            result.append(norm_image)
        
        ds = open(val_dataset_file)
        lines = ds.readlines()
        datum = lines[i]
        video_path = datum.replace('\n', '')
        frames_path = video_path + '.mp4'
        ret = show_heatmap_video(frames_path, result, cnt)
        if ret:
            print("Successful video export!\n\n")
        cnt += 1


if __name__ == "__main__":
    main()
