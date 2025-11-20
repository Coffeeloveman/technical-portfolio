# 3️⃣ 데이터 활용 및 분석 : **머신러닝 라이브러리를 이용한 재현 가능한 개발 결과물 공개 여부**
---

## Summary
본 연구의 모델 학습 및 평가 파이프라인은 **PyTorch** 기반으로 구현되었으며, 모델 **추론**, **평가**까지 전 과정을 자동화했습니다.  
실험 재현을 위해 환경 구축 방법을 [2_setup](https://github.com/Coffeeloveman/technical-portfolio/tree/main/2_setup)을 공개합니다.
이를 통해 사용자는 동일한 환경을 구성한 뒤, 제공된 명령어만으로 추론 및 평가를 동일하게 재현할 수 있습니다. 또한, 실험에 사용된 **하이퍼파라미터** 등을 코드 내 설정에 명시 및 추론에 필요한 데이터셋을 공개하여 **결과 재현성(Reproducibility)** 을 보장했습니다.  
---

## 폴더 구성
- datasets
  - coco_30k_10k.csv
  - i2p.csv
  - mma-diffusion-nsfw-adv-prompts.csv
- infer
  - adv_samping.py
  - coco_sampling.py
  - unlearncanvas-sampling,py
- eval
  - clip_similarity.py
  - compute_nudity_rate.py
  - fid.py
  - nudity_eval.py
  - unlearncanvas-acc-result.py
  - unlearncanvas-acc.py
--- 

## 추론 방법
- requirement
  - [1_language](https://github.com/Coffeeloveman/technical-portfolio/tree/main/1_language)에서 학습한 text encoder weight

