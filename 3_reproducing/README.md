# 📘 [평가 코드물](https://github.com/Coffeeloveman/RMU-SD/blob/main/Implement.md)
위 링크에는 각 코드 실행을 위한 환경 구축 파일 및 세부 설명 및 사용 방법이 담겨 있습니다.
---

## 1️⃣ 데이터사이언스 공통  
### **Python, SQL, R, Java, Scala, Go, C/C++, Javascript 등 데이터 처리 언어 활용 능력**

본 연구에서는 **Python**을 주요 개발 언어로 사용하여 모델 **훈련**, **추론**, **평가** 코드를 작성하였습니다.  

> **Languages:** Python 100.0%

---

## 2️⃣ 데이터사이언스 공통  
### **Linux, Docker, Virtual Machines, Kubernetes 등을 활용한 데이터 활용 및 분석을 위한 환경 구축 여부**

본 연구의 실험은 **Ubuntu 22.04 LTS (Linux)** 환경에서 수행되었습니다. 도커 환경의 경우 [docker_env](https://github.com/Coffeeloveman/RMU-SD/blob/main/Docker.md)를 통해 구축했습니다.
또한 모델 학습 및 추론, 평가에 필요한 환경을 위해 패키지 의존성을 명시한 환경 설정 파일(`environment.yml`)을 구축하였으며, 이를 통해 **Conda 기반 가상환경**을 생성하여 최종 실행 환경을 구성하였습니다. 

---

## 3️⃣ 데이터 활용 및 분석  
### **머신러닝 라이브러리를 이용한 재현 가능한 개발 결과물 공개 여부**

본 연구의 모델 학습 및 평가 파이프라인은 **PyTorch** 기반으로 구현되었으며, 모델 **학습**, **추론**, **평가**까지 전 과정을 자동화했습니다.  
실험 재현을 위해 모든 코드와 스크립트를 정리한 [공개 저장소](https://github.com/Coffeeloveman/RMU-SD)와 환경 설정 파일(`environment.yml`)을 함께 제공했습니다.  

이를 통해 사용자는 동일한 Conda 환경을 구성한 뒤, 제공된 명령어만으로 학습 및 평가를 동일하게 재현할 수 있습니다. 또한, 실험에 사용된 **하이퍼파라미터** 등을 코드 내 설정에 명시하여 **결과 재현성(Reproducibility)** 을 보장했습니다.  
본 결과물은 머신러닝 프레임워크(**PyTorch**)와 공개 환경 설정을 기반으로 하여,  동일 플랫폼(**Ubuntu 22.04**)에서 완전한 재현이 가능합니다.
