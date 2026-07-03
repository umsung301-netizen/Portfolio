import os
import cv2
import glob
import torch
import pickle
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

def find_valid_dataset(base_dir='/data/nas/wallpaper_imgs/photo_admin'):
    """이미지·마스크·메타가 모두 존재하는 유효 샘플 탐색 (rel_dir 중첩 경로 대응)"""
    meta_dir = f"{base_dir}_meta/meta"
    mask_dir = f"{base_dir}_meta/mask"
    pkl_files = glob.glob(f"{meta_dir}/**/*.pkl", recursive=True)

    for pkl_path in sorted(pkl_files):
        base_name = os.path.splitext(os.path.basename(pkl_path))[0]
        rel_dir = os.path.relpath(os.path.dirname(pkl_path), meta_dir)
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        segments_info = data.get("segments_info", [])
        if len(segments_info) < 3:
            continue

        src = data.get("source_image") or (segments_info[0].get("source_image") if segments_info else None) or f"{base_name}.jpg"
        img_dir = os.path.join(base_dir, rel_dir) if rel_dir != "." else base_dir
        msk_dir = os.path.join(mask_dir, rel_dir) if rel_dir != "." else mask_dir
        img_path = os.path.join(img_dir, src)
        if not os.path.exists(img_path):
            stem = os.path.splitext(src)[0]
            for ext in (".jpg", ".png", ".jpeg"):
                cand = os.path.join(img_dir, stem + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
        mask_path = os.path.join(msk_dir, f"{base_name}.pt")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            return img_path, mask_path, pkl_path, segments_info

    return None, None, None, None

def main():
    image_path, mask_path, pkl_path, segments_info = find_valid_dataset()
    if not image_path:
        print("Valid dataset not found.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    depth_model = DepthAnythingV2(**model_configs['vitl'])
    ckpt_path = "checkpoints/depth_anything_v2_vitl.pth"
    depth_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    depth_model = depth_model.to(device).eval()

    original_img = cv2.imread(image_path)
    panoptic_map = torch.load(mask_path).numpy()

    with torch.no_grad():
        depth_map = depth_model.infer_image(original_img)

    results = []
    for seg in segments_info:
        seg_id = seg["id"]
        mask = (panoptic_map == seg_id)
        
        if not np.any(mask):
            continue
            
        median_depth = np.median(depth_map[mask])
        results.append({
            "seg_id": seg_id,
            "category_id": seg.get("category_id", -1),
            "area": seg.get("area", np.sum(mask)),
            "depth": median_depth
        })

    # 오름차순 정렬 (값이 작을수록 배경)
    results.sort(key=lambda x: x["depth"], reverse=False)

    print(f"Dataset: {os.path.basename(image_path)}")
    print("-" * 55)
    print(f"{'Order':<6} | {'Seg ID':<6} | {'Cat ID':<6} | {'Area':<8} | {'Depth'}")
    print("-" * 55)
    for idx, res in enumerate(results):
        print(f"[{idx + 1:02d}]    | {res['seg_id']:<6} | {res['category_id']:<6} | {res['area']:<8} | {res['depth']:.2f}")
    print("-" * 55)

if __name__ == '__main__':
    main()
