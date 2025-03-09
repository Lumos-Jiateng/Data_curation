export CUDA_VISIBLE_DEVICES=5
python pipeline.py \
  --config /shared/nas/data/m1/jiateng5/Data_curation/MotionSegCap/third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /shared/nas/data/m1/jiateng5/Data_curation/MotionSegCap/third_party/Grounded-Segment-Anything/pretrained_checkpoints/groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint /shared/nas/data/m1/jiateng5/Data_curation/MotionSegCap/third_party/Grounded-Segment-Anything/pretrained_checkpoints/sam_hq_vit_h.pth \
  --use_sam_hq \
  --use_sam2 \
  --input_dataset_name ssv2 \
  --output_dir "outputs/" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --device "cuda"