import os
import glob
import json
import re
import torch
from PIL import Image
import google.generativeai as genai
from pydantic import BaseModel, Field
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------
# 1. 스키마 정의 (JSON 강제화)
# ---------------------------------------------------------
class ImageMetadata(BaseModel):
    title: str = Field(
        description="이미지를 대표하는 짧고 직관적인 제목 (한국어, 20자 이내)"
    )
    description: str = Field(
        description="이미지의 상황, 분위기, 주요 객체를 포함한 상세 설명 (한국어, 2문장 내외)"
    )
    tags: list[str] = Field(description="이미지의 핵심 키워드 5개 (배열 형태)")
    category: str = Field(
        description="다음 중 가장 알맞은 카테고리 1개: [풍경, 인물, 음식, 인테리어, IT기기, 기타]"
    )


# ---------------------------------------------------------
# 2. Qwen 로컬 모델 로드 (Global)
# ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Qwen 모델 로딩 중... (VRAM 용량에 따라 시간이 걸릴 수 있습니다)")
processor = AutoProcessor.from_pretrained(MODEL_ID)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    quantization_config=quantization_config, 
    device_map="auto"
)


# ---------------------------------------------------------
# 3. 모델별 파이프라인 함수
# ---------------------------------------------------------
def process_image_qwen(image_path: str):
    """Qwen 모델을 활용한 메타데이터 추출"""
    image = Image.open(image_path).convert("RGB")
    schema_json = ImageMetadata.schema_json()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"주어진 이미지를 분석하여 다음 JSON 스키마에 정확히 맞춰서 답변해 줘.\n설명이나 인사말 등 다른 텍스트는 절대 출력하지 말고 오직 JSON 형식만 출력해.\nSchema:\n{schema_json}",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(model.device)

    # 모델 추론
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=512, temperature=0.2, do_sample=True
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]
    generated_text = processor.decode(generated_tokens, skip_special_tokens=True)

    # 후처리 (JSON 파싱 및 검증)
    try:
        json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            parsed_data = ImageMetadata.parse_raw(clean_json)
            return parsed_data.dict()
        else:
            raise ValueError("JSON 형식을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파싱 에러: {e}\n원본 텍스트:\n{generated_text}")
        return None


def process_image_gemini(image_path: str, api_key: str):
    """Gemini API를 활용한 메타데이터 추출 (예비용)"""
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        img = Image.open(image_path)
        response = gemini_model.generate_content(
            ["주어진 이미지를 분석하여 스키마에 맞게 메타데이터를 추출해 줘.", img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ImageMetadata,
                temperature=0.2,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini API 에러 발생: {e}")
        return None


# ---------------------------------------------------------
# 4. 메인 실행 블록
# ---------------------------------------------------------
if __name__ == "__main__":
    # [보안 픽스 완료] API 키는 코드에 노출하지 않고 환경변수에서만 불러옵니다.
    # (Qwen만 사용할 경우 이 값은 None이어도 정상 작동합니다.)
    API_KEY = os.environ.get("GEMINI_API_KEY")

    # 입출력 폴더 경로 설정
    INPUT_DIR = "/content/drive/MyDrive/huvle"
    OUTPUT_DIR = "/content/drive/MyDrive/huvle/metadata_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_images = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    target_images = all_images[:100]

    print(f"전체 파일 {len(all_images)}개 중 {len(target_images)}개의 메타데이터 추출을 시작합니다.\n")

    # 순회하며 추출 및 저장
    for idx, img_path in enumerate(target_images, start=1):
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        json_save_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")

        # 이미 추출된 JSON이 있다면 건너뛰기
        if os.path.exists(json_save_path):
            print(f"[{idx}/{len(target_images)}] ⏩ {base_filename}.json 파일이 이미 존재하여 건너뜁니다.")
            continue

        print(f"[{idx}/{len(target_images)}] ⏳ {img_path} 분석 중...")

        # Gemini 대신 Qwen 로컬 모델 사용
        result_data = process_image_qwen(img_path)

        if result_data is not None:
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
            print(f"  -> ✅ 저장 완료: {json_save_path}")
        else:
            print(f"  -> ❌ 에러 발생: 분석을 실패하여 건너뜁니다.")

    print("\n🎉 모든 작업이 완료되었습니다!")
