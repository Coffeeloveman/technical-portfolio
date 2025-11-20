
## 🐳 CUDA 이미지 가져오기

```
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```
 
## 🔨 이미지 빌드
```
docker build -t name --build-arg UNAME=name --build-arg UID=your_uid --build-arg GID=your_gid .
```
## ▶️ 컨테이너 실행
```
docker run --gpus all -i -t -d -u $(id -u):$(id -g) -v host_path:container_path -v /etc/localtime:/etc/localtime -e DISPLAY=$DISPLAY --ipc=host --name container_name -id image_name /bin/bash
```
