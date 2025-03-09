import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
import ast

# change to your path to install grounded-sam
sys.path.append("/shared/nas/data/m1/jiateng5/Data_curation/MotionSegCap/third_party/Grounded-Segment-Anything")

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

from video_datasets import SSV2_Dataset

# adding sam2
from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "/shared/nas/data/m1/jiateng5/dcore_trackgpt/trackgpt/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

import os
import numpy as np
import cv2
from PIL import Image
#from moviepy.video.io.VideoFileClip import VideoFileClip
#from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import io
import base64
import json
from PIL import Image
from openai import OpenAI

# 1. Create the OpenAI client instance here
client = OpenAI(api_key="")

import re

def encode_pil_image(pil_image: Image.Image) -> str:
    """
    Encodes a PIL Image into a base64 string,
    suitable for sending to the OpenAI API via data URIs.
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_objects_for_action(
    action_desc: str,     # String describing the action, e.g. "holding a cup in front of another cup"
    frame1: Image.Image,  # First sampled PIL image
    frame2: Image.Image   # Second sampled PIL image
) -> list:
    """
    1) Converts two PIL images to base64-encoded data URIs.
    2) Sends them along with the action description to the GPT-4o model (via `client`).
    3) Instructs GPT-4o to return a JSON array of relevant objects.
    4) Parses and returns that JSON array as a Python list.
    """

    # 2. Encode the images
    base64_image1 = encode_pil_image(frame1)
    base64_image2 = encode_pil_image(frame2)

    # 3. Prepare the messages payload
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Action description: {action_desc}.\n"
                        "Identify all the objects associated with this action description and exists within the following images. Do not include the background objects into your answer."
                        "Return only a list of the names objects, "
                        "like ['arm', 'bottle', 'television']."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"},
                },
            ],
        }
    ]

    # 4. Call GPT-4o using the sealed `client`
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Extract the text returned by the model
    raw_response = response.choices[0].message.content

    # Attempt to parse the response as JSON
    def parse_list_from_brackets(raw_response: str) -> list:
        """
        Uses a regular expression to capture everything from the first '[' 
        to the last ']' in raw_response, then tries to parse it as JSON.
        Returns a Python list if successful, or [] otherwise.
        """
        # This pattern captures everything from the first '[' through the final ']'
        pattern = r"\[.*\]"
        match = re.search(pattern, raw_response, flags=re.DOTALL)
        if not match:
            return []

        substring = match.group(0)  # e.g. ["hand", "cup", "mug"]
        
        # Attempt to parse that substring as JSON
        try:
            parsed_list = ast.literal_eval(substring)
            if isinstance(parsed_list, list):
                return parsed_list
            return []
        except json.JSONDecodeError:
            return []
    
    result = parse_list_from_brackets(raw_response)
    return result
    '''
    try:
        objects_list = json.loads(raw_response)
    except json.JSONDecodeError:
        objects_list = []

    return objects_list
    '''



def save_source(pil_image_list, output_path, fps=10):
    """
    Convert a list of PIL images into an MP4 video.

    Args:
        image_list (list): List of PIL Image objects.
        output_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the video.
    """
    # Convert the first image to a numpy array to determine the frame size.
    first_frame = np.array(pil_image_list[0])
    
    # Check for grayscale images (2D array) and adjust shape if necessary.
    if first_frame.ndim == 2:
        height, width = first_frame.shape
        channels = 1
    else:
        height, width, channels = first_frame.shape

    # Define the codec and create VideoWriter object.
    # 'mp4v' codec is widely compatible with .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in pil_image_list:
        # Convert the PIL image to a numpy array.
        frame = np.array(image)
        
        # If the image is grayscale, convert to BGR format.
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            # If the image has an alpha channel (RGBA), convert to BGR.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            # Convert RGB to BGR (OpenCV uses BGR order).
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        video_writer.write(frame)

    # Release the video writer.
    video_writer.release()
    print(f"Video saved to {output_path}")


def save_mask(pil_image_list, video_segments, output_path, fps=10):
    """
    Save the masked video by applying the masks from video_segments.

    Args:
        pil_image_list (list): List of PIL images.
        video_segments (dict): Dictionary containing masks for each frame.
        output_path (str): Path to save the masked video.
        fps (int): Frames per second of the output video.
    """
    frame_array = []

    for frame_idx, img in enumerate(pil_image_list):
        frame = np.array(img)  # Convert PIL image to numpy array
        mask_dict = video_segments.get(frame_idx, {})  # Get masks for this frame

        combined_mask = np.zeros((256, 256), dtype=np.uint8)  # Initialize combined mask
        for _, mask in mask_dict.items():
            combined_mask = np.logical_or(combined_mask, mask.squeeze()).astype(np.uint8) * 255  # Merge masks

        # Apply mask: Keep original image where mask is present, black out the rest
        masked_frame = np.zeros_like(frame)
        for i in range(3):  # Apply mask to all RGB channels
            masked_frame[:, :, i] = frame[:, :, i] * (combined_mask // 255)

        frame_array.append(masked_frame)

    height, width, _ = frame_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frame_array:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Masked video saved at {output_path}")


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument(
        "--use_sam2", action="store_true", help="using sam-hq for prediction"
    )
    # change into input_dataset_name
    parser.add_argument("--input_dataset_name", type=str, required=True, help="path to image file")
    #parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    use_sam2 = args.use_sam2
    #image_path = args.input_image # input_image -> dataset
    #text_prompt = args.text_prompt # text_prompt -> get from dataset
    if args.input_dataset_name == 'ssv2':
        dataset = SSV2_Dataset()
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path
    
    #import ipdb;ipdb.set_trace()
    
    if use_sam_hq:
        if use_sam2: 
            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
            
        else:
            predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    # get each piece of data
    for i in range(len(dataset)):
        pil_image_list = dataset[i]['pil_image_list']
        tensored_image_list = dataset[i]['tensored_image_list']
        original_question = dataset[i]['question']
        original_label = dataset[i]['answer']
        
        video_path = dataset[i]['video_path']
        unique_id = os.path.splitext(os.path.basename(video_path))[0]
    
    ####### Utilize the first image for detecting all objects involved in the action #######
        
        # load image
        image_pil, image = pil_image_list[0],tensored_image_list[0]
        # load model
        model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
 
        text_prompt = original_label
        import ipdb;ipdb.set_trace()
        text_prompt_list = get_objects_for_action(text_prompt, pil_image_list[0],pil_image_list[8])
        # Use GPT-models to curate the text_prompt to associated objects. 
        
        text_prompt_list = ['hand','cup','coffee cup']
        
        boxes_list = []
        
        for text_prompt in text_prompt_list:
        # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, device=device
            )
            boxes_list.append(boxes_filt)
        boxes_filt = torch.cat(boxes_list, dim=0)
        
        image_array = np.array(image_pil)
        
        import ipdb;ipdb.set_trace()
        
        inference_state = predictor.init_state(video_path=pil_image_list)
        predictor.reset_state(inference_state)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        
        boxes_array_list = boxes_filt.tolist()
        ann_frame_idx = 0 
        
        for ann_obj_id, box in enumerate(boxes_array_list):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
            )
            
        video_segments = {} 
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }  
       
        # draw output image
        #if if_vis:
            
            # this part can save as videos for visualization
            '''
            vis_frame_stride = 1
            for out_frame_idx in range(0, len(pil_image_list), vis_frame_stride):
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {out_frame_idx}")
                plt.imshow(pil_image_list[out_frame_idx])
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    show_mask(out_mask.astype(int), plt.gca(), random_color=True) 
                    
                plt.savefig(f'/shared/nas/data/m1/jiateng5/MotionSegCap/pipelines/debug_vis/{out_frame_idx}.jpg')
            '''
        # store with the unique video_id 
        root_dir = os.path.join('/shared/nas/data/m1/jiateng5/Data_curation/MotionSegCap/curated_data',args.input_dataset_name)
        store_dir = os.path.join(root_dir,unique_id)
        initial_detect_path = os.path.join(store_dir,'initial_detect.jpg')
        original_video_path = os.path.join(store_dir,'source.mp4')
        masked_video_path = os.path.join(store_dir,'mask.mp4')
        
        os.makedirs(store_dir,exist_ok = True)
            
        for i in range(1):
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(pil_image_list[i])
            for box in boxes_array_list:
                show_box(box, plt.gca(),text_prompt)
        #show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        
            plt.savefig(initial_detect_path)
            plt.close()
        
        # draw out image
        import ipdb;ipdb.set_trace()
        
        save_source(pil_image_list, original_video_path)
        save_mask(pil_image_list, video_segments, masked_video_path)
        
        
        
        '''
        plt.figure(figsize=(10, 10))
        plt.imshow(image_array)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "grounded_sam_output.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
        
        import ipdb;ipdb.set_trace()
        '''
