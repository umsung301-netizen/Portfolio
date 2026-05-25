# 1. 가상환경 활성화
source ./venv/bin/activate

# 2. 입출력 폴더 설정 (본인 환경에 맞게 경로를 수정하세요)
INPUT_DIR="./data/input_images"
OUTPUT_DIR="./data/output_meta"

# 3. 모델 실행
# GPU 번호는 환경에 맞게 CUDA_VISIBLE_DEVICES=0 등으로 조절하세요.
CUDA_VISIBLE_DEVICES=0 python extract.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --config-file "configs/eov_seg/eov_seg_convnext_l.yaml" \
  --opts MODEL.WEIGHTS "weights/convnext-l.pth"
