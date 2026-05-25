import json
from PIL import Image
import google.generativeai as genai
from pydantic import BaseModel, Field
import torch
import re
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
)

# 배경화면 입력으로 받아서 제목, 설명, 태크, 카테고리 뽑는 코드, Gemini API를 Qwen으로 대체("Qwen/Qwen2.5-VL-7B-Instruct")


# ---------------------------------------------------------
# 1. 스키마 정의 (나중에 로컬 Llama로 바꿀 때도 100% 동일하게 사용)
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


# # ---------------------------------------------------------
# # 2. Hugging Face 모델 및 프로세서 로드
# # ---------------------------------------------------------
# # 주의: 실제 사용하려는 Llama 비전 모델의 Hugging Face ID로 변경하세요.
# # MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" # (예시 ID)
# # MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# print("모델 로딩 중... (VRAM 용량에 따라 시간이 걸릴 수 있습니다)")
# processor = AutoProcessor.from_pretrained(MODEL_ID)

# # bfloat16을 사용하여 VRAM 메모리 사용량을 절반으로 줄입니다.
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     device_map="auto",          # 사용 가능한 GPU에 자동 할당
#     torch_dtype=torch.bfloat16  # 16비트 정밀도 연산
# )


def process_image_hf(image_path: str):
    # 이미지 로드 및 RGB 변환
    image = Image.open(image_path).convert("RGB")

    # ---------------------------------------------------------
    # 3. 프롬프트 구성 (JSON 생성 강제)
    # ---------------------------------------------------------
    # Hugging Face 로컬 추론에서는 API처럼 response_format을 바로 넘길 수 없으므로,
    # 프롬프트에 스키마를 강하게 주입하여 JSON만 뱉도록 유도해야 합니다.
    schema_json = ImageMetadata.schema_json()

    # Llama의 챗 템플릿(Chat Template) 형식에 맞춰 프롬프트 작성
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
<image>
주어진 이미지를 분석하여 다음 JSON 스키마에 정확히 맞춰서 답변해 줘.
설명이나 인사말 등 다른 텍스트는 절대 출력하지 말고 오직 JSON 형식만 출력해.
Schema:
{schema_json}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    # 프로세서를 통해 이미지와 텍스트를 모델이 이해할 수 있는 텐서(Tensor)로 인코딩
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # ---------------------------------------------------------
    # 4. 모델 추론
    # ---------------------------------------------------------
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,  # 환각 최소화
            do_sample=True,  # temperature 적용을 위해 True
        )

    # 입력 프롬프트 길이를 제외하고 새로 생성된 텍스트만 디코딩
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]
    generated_text = processor.decode(generated_tokens, skip_special_tokens=True)

    # ---------------------------------------------------------
    # 5. 후처리 (JSON 파싱 및 검증)
    # ---------------------------------------------------------
    try:
        # 모델이 코드 블록(```json ... ```)을 포함해서 뱉을 경우를 대비해 정규식으로 추출
        json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            parsed_data = ImageMetadata.parse_raw(
                clean_json
            )  # Pydantic으로 스키마 검증
            return parsed_data.dict()
        else:
            raise ValueError("JSON 형식을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파싱 에러: {e}\n원본 텍스트:\n{generated_text}")
        return None


# ---------------------------------------------------------
# 2. 임시 API 파이프라인 엔진 (Gemini)
# ---------------------------------------------------------
def process_image_gemini(image_path: str, api_key: str):
    # Gemini API 세팅
    genai.configure(api_key=api_key)
    # 가볍고 빠른 flash 모델 (무료 티어로 충분히 테스트 가능)
    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        # Base64 변환 없이 PIL Image 객체를 그대로 던질 수 있어 코드가 더 깔끔합니다.
        img = Image.open(image_path)

        # Pydantic 스키마를 던져서 JSON 출력을 완벽히 강제합니다.
        response = model.generate_content(
            ["주어진 이미지를 분석하여 스키마에 맞게 메타데이터를 추출해 줘.", img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ImageMetadata,  # Pydantic 모델 주입
                temperature=0.2,
            ),
        )

        # Gemini가 뱉은 JSON 텍스트를 파이썬 딕셔너리로 파싱해서 반환
        return json.loads(response.text)

    except Exception as e:
        print(f"Gemini API 에러 발생: {e}")
        return None


# ---------------------------------------------------------
# (qwen)
# ---------------------------------------------------------
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print("모델 로딩 중")
processor = AutoProcessor.from_pretrained(MODEL_ID)

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=quantization_config, device_map="auto"
)


def process_image_qwen(image_path: str):
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

    # 모델 추론 (기존 Llama 코드와 동일)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=512, temperature=0.2, do_sample=True
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]
    generated_text = processor.decode(generated_tokens, skip_special_tokens=True)

    # 후처리 (기존 Llama 코드와 완전히 동일)
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

if __name__ == "__main__":
    import os
    import glob
    import json

    # 1. API 키 준비
    # API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyADiYo_qEf-wEcQAw2TyWkhYFL0_qyFQ6M")

    # 2. 입출력 폴더 경로 설정
    INPUT_DIR = "/content/drive/MyDrive/huvle"
    OUTPUT_DIR = "/content/drive/MyDrive/huvle/metadata_results"

    # 저장할 폴더가 없으면 자동으로 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. 폴더 내 모든 jpg 파일 목록 가져오기
    all_images = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))

    # 딱 100개만 슬라이싱하여 타겟으로 설정
    target_images = all_images[:100]

    print(
        f"전체 파일 {len(all_images)}개 중 {len(target_images)}개의 메타데이터 추출을 시작합니다.\n"
    )

    # 4. 순회하며 추출 및 저장
    for idx, img_path in enumerate(target_images, start=1):
        # 파일명만 추출 (예: "image_01.jpg" -> "image_01")
        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # 저장될 JSON 파일의 최종 경로 (예: "data/metadata_results/image_01.json")
        json_save_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")

        # [선택] 이미 추출된 JSON이 있다면 건너뛰기 (중단 후 재실행 시 유용)
        if os.path.exists(json_save_path):
            print(
                f"[{idx}/{len(target_images)}] ⏩ {base_filename}.json 파일이 이미 존재하여 건너뜁니다."
            )
            continue

        print(f"[{idx}/{len(target_images)}] ⏳ {img_path} 분석 중...")

        # ⭐️ 기존 파이프라인 클래스 대신 임시 API 함수 호출
        # result_data = process_image_gemini(img_path, API_KEY)
        result_data = process_image_qwen(img_path)

        # 함수가 정상적으로 데이터를 반환했다면 (None이 아니라면)
        if result_data is not None:
            # 5. 결과를 JSON 파일로 저장
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
            print(f"  -> ✅ 저장 완료: {json_save_path}")
        else:
            print(f"  -> ❌ 에러 발생: 분석을 실패하여 건너뜁니다.")

    print("\n🎉 모든 작업이 완료되었습니다!")
