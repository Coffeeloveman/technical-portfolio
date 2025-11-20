
## 2️⃣ 데이터사이언스 공통  
### **Linux, Docker, Virtual Machines, Kubernetes 등을 활용한 데이터 활용 및 분석을 위한 환경 구축 여부**

---
### 설명
본 연구의 모든 실험은 **Ubuntu 22.04 LTS (Linux)** 환경에서 수행되었습니다.  
GPU 활용하기 위해 **NVIDIA CUDA Docker 이미지**를 기반으로 컨테이너 환경을 구성했습니다.
또한, [1_language](https://github.com/Coffeeloveman/technical-portfolio/tree/main/1_language), [3_reproducing](https://github.com/Coffeeloveman/technical-portfolio/tree/main/3_reproducing)에 필요한 환경을 위해 패키지 의존성을 명시한 환경 설정 파일(`environment.yml`)을 구축하였으며, 이를 통해 **Conda 기반 가상환경**을 생성하여 최종 실행 환경을 구성하였습니다. 

---
### Summary
- 운영체제: **Ubuntu 22.04 LTS**
- GPU 환경: **NVIDIA CUDA 11.8 + cuDNN 8**
- 컨테이너: **Docker**
- 가상환경: **Conda (`environment.yml` 기반)**
---

## 환경 구축 과정

### 🐳 CUDA 이미지 가져오기

```
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```
 
### 🔨 이미지 빌드
```
docker build -t name --build-arg UNAME=name --build-arg UID=your_uid --build-arg GID=your_gid .
```
### ▶️ 컨테이너 실행
```
docker run --gpus all -i -t -d -u $(id -u):$(id -g) -v host_path:container_path -v /etc/localtime:/etc/localtime -e DISPLAY=$DISPLAY --ipc=host --name container_name -id image_name /bin/bash
```


### Conda 환경 구축
```
conda env create -f environment.yml
conda activate <env_name>
```
---
