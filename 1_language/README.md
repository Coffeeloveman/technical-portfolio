## 1️⃣ 데이터사이언스 공통: **Python 데이터 처리 언어 활용 능력**

---

## 프로젝트 개요

본 연구에서는 **Python**을 주요 개발 언어로 사용하여  
Text-to-Image 모델인 **Stable Diffusion v1.4 (SD1.4)**의 **텍스트 인코더(CLIP Text Encoder)**를 직접 학습·수정하는 코드를 구현했습니다.

목표는 SD1.4가 생성하는 이미지 중,
- 저작권 이슈가 있는 **특정 스타일(style)**  
- 노출이 포함된 **nudity와 같은 NSFW 개념**

과 같이 **원치 않는/위험한 개념을 포함한 이미지 생성을 억제**하는 것입니다.

이를 위해, SD1.4에서 프롬프트를 통해 이미지를 생성할 때 사용되는  
**프롬프트의 텍스트 표현(embedding)을 담당하는 텍스트 인코더를 직접 재학습하여,  
지우고자 하는 개념의 표현을 “무의미한 랜덤 벡터”로 매핑**하도록 설계하였습니다.

---

## 학습 목표

- **주요 언어:** Python  
- **대상 모델:** Stable Diffusion v1.4의 CLIP Text Encoder  
- **핵심 아이디어:**  
  - 특정 개념(예: `"nudity"`, `"van gogh"`)에 해당하는 텍스트 인코더의 representation을  
    **랜덤 벡터로 치환되도록 학습**하여,  
    해당 개념이 프롬프트에 포함되더라도 **모델이 제대로 조건을 인식하지 못하게 만드는 것**입니다.
  - 이를 통해, 원본 모델의 표현력은 유지하되, **지우고 싶은 개념만 선택적으로 약화/제거**하는 것을 목표로 합니다.

---

## 학습 방법 요약

지우고자 하는 개념의 텍스트 인코더 출력을  
**임의의(random) 타깃 벡터**로 매핑하도록 하는 **L2 loss 기반 회귀 학습**을 수행합니다.

- 입력: 특정 개념이 포함된 텍스트 프롬프트 (예: `"a nude person"`, `"in the style of van gogh"`)
- 출력: 텍스트 인코더의 embedding 벡터
- 목표:  
  - 원래의 embedding 대신, 미리 샘플링한 **랜덤 벡터**에 가깝도록 L2 loss로 학습
  - 이로 인해, 해당 개념은 “일관된 의미 있는 표현”을 잃고,  
    이미지 생성 시 **조건으로서의 역할을 거의 하지 못하게** 됩니다.

---

## 코드 구성

- `environment.yml`  
  - 학습을 위한 Python 환경 및 필요한 라이브러리 의존성 목록  
  - `conda env create -f environment.yml`로 재현 가능한 환경 구성 가능

- `save_pipe.py`  
  - Hugging Face의 `CompVis/stable-diffusion-v1-4` 파이프라인에서  
    **텍스트 인코더를 추출 및 저장**하는 스크립트

- `train.py`  
  - 저장된 텍스트 인코더를 로드하여 **특정 개념을 지우도록 파인튜닝**하는 메인 학습 코드
  - 주요 기능:
    - `target_concept`에 대응하는 텍스트 인코더 representation을  
      **랜덤 벡터로 매핑**되도록 L2 loss로 학습
    - `concept_type`에 따라 서로 다른 **learning rate / 하이퍼파라미터** 적용 가능  
      (예: `style`, `object`, `nsfw` 등)

- `utils.py`  
  - 학습에 필요한 공통 함수 모음 (예: optimizer 생성)

---

## 학습 과정
1. conda 환경 구축
```
conda env create -f environment.yml
conda activate <env_name>
```
2. text encoder 저장
```
python save_pipe.py --pipeline "CompVis/stable-diffusion-v1-4" --sd_version "sd-14"
```
3. 학습
```
python train.py --target_concept "nudity" --concept_type 'nsfw' --epoch 50 --text_encoder_path "models/sd-14/text_encoder" --save_path '/save_dir'
python train.py --target_concept "van gogh" --concept_type 'style' --epoch 50 --text_encoder_path "models/sd-14/text_encoder" --save_path '/save_dir'
```
- target_concept
  - 지우고자하는 개념
- concept_type
  - target concept이 속한 도메인 (eg., style, object, nsfw)으로 도메인에 따라 Learning Rate를 다르게 적용
- epoch  
  - 학습 에폭 수 (데이터/개념 난이도에 따라 조정 가능)
- save_path
  - 학습된 text encoder weight 저장 경로

## 학습 결과 및 활용

- 원래 **SD1.4 파이프라인**의 텍스트 인코더를  
  학습된(**text-encoder-edited**) 텍스트 인코더로 **교체**한 뒤,
- `target_concept`가 포함된 프롬프트로 이미지를 생성하면,

다음과 같은 효과를 확인할 수 있습니다.

- 생성되는 이미지에서 해당 개념이 **거의 드러나지 않거나**
- 개념의 시각적 특징/의미가 **크게 약화**된 것을 시각적으로 확인 가능

추론 및 정량/정성 평가 방법, 재현 스크립트 등은 아래 디렉터리에 정리되어 있습니다.

- [**3_reproducing**](https://github.com/Coffeeloveman/technical-portfolio/tree/main/3_reproducing)
  - 개념 제거 전/후 이미지 비교
  - 다양한 프롬프트 설정 예시
  - 재현 가능한 평가 실험 코드

---

## 기술 스택

> **Languages:** Python 100.0%

### Main frameworks & libraries

- **PyTorch**
- **diffusers**
- **transformers**

### 환경 관리

- **Conda**
  - `environment.yml` 기반으로 재현 가능한 실험 환경 구성


---


