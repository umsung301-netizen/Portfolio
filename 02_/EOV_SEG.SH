# source ./venv/bin/activate
input_dir="/data/nas/wallpaper_imgs/photo_admin"
output_dir="/data/nas/wallpaper_imgs/photo_admin_meta"
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/eov-seg/bin/python extract.py --input "$input_dir" --output "$output_dir" --config-file "configs/eov_seg/eov_seg_convnext_l.yaml" --opts MODEL.WEIGHTS weights/convnext-l.pth