import transformers
from torch.utils.data import Dataset
import logging
import json
from typing import Dict, Optional, Sequence
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
import random

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch

# always sample K frames of shape 3,256,256
def sample_frames_from_videos(video_path, K):
    """
    Samples K frames from an input video and returns:
    1. A list of PIL images (all resized to 256x256).
    2. A list of corresponding transformed tensors (3x256x256).

    If the video has fewer than K frames, it duplicates frames to match K.
    If the video has more than K frames, it samples frames uniformly.

    Parameters:
    - video_path (str): Path to the input video file.
    - K (int): Number of frames to sample.

    Returns:
    - List of K sampled frames as PIL images (all resized to 256x256).
    - List of K sampled frames as transformed tensors (shape: 3x256x256).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError("Failed to read video or empty video file.")

    # Generate indices for sampling K frames
    if total_frames <= K:
        indices = np.linspace(0, total_frames - 1, K, dtype=int)  # Duplicate frames if needed
    else:
        indices = np.linspace(0, total_frames - 1, K, dtype=int)  # Uniform sampling

    transform = T.Compose([
        T.Resize((256, 256)),  # Resize to 256x256
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pil_images = []
    tensor_images = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame at index {idx}.")
        
        # Convert BGR (OpenCV) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Resize PIL image to 256x256
        pil_image_resized = pil_image.resize((256, 256), Image.BILINEAR)

        # Apply transformation (resize, convert to tensor, normalize)
        tensored_image = transform(pil_image)

        pil_images.append(pil_image_resized)
        tensor_images.append(tensored_image)

    cap.release()
    return pil_images, tensor_images

class SSV2_Dataset(Dataset):
    
    def __init__(self, json_file = '/home/qjx0814/ecole-video/data_construction/instruction_data/qwen-SSV2-train-32B-processed.json', root_dir="/shared/nas/data/m1/shared-resource/vision-language/data/raw/ssv2/videos/clips_downsampled_5fps_downsized_224x224", num_frames = 16):
        
        with open(json_file,'r') as f:
            content = json.load(f)
        
        print(f"ssv2 length: {len(content)}") # 107167
        
        self.num_frames = num_frames     
        # extract data_piece.
        self.video_path_list = []
        self.question_list = []
        self.answer_list = []
        
        for i in range(0,len(content),1):
            video_id = content[i]['id']
            '''
            for j in range(len(content[i]['instruction_data'])):
                question = content[i]['instruction_data'][j]['Question']
                answer = content[i]['instruction_data'][j]['Answer']
                self.video_path_list.append(root_dir + '/' + video_id+'.mp4')
                self.label = content[i]['label']
            '''
            # also add another question 
            action_question = "What's the key action in the video?"
            action_answer = content[i]['label']
            self.video_path_list.append(root_dir + '/' + video_id+'.mp4')
            self.question_list.append(action_question)
            self.answer_list.append(action_answer)
    
    def __getitem__(self,i):         
        video_path = self.video_path_list[i]
        pil_image_list,tensored_image_list = sample_frames_from_videos(video_path, self.num_frames)
        return dict(pil_image_list= pil_image_list, tensored_image_list=tensored_image_list, question= self.question_list[i],answer=self.answer_list[i])
    
    def __len__(self):
        return len(self.video_path_list)
    
if __name__ == "__main__":
    dataset = SSV2_Dataset()
    import ipdb;ipdb.set_trace()