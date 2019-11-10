YOLO V3
===

The main idea of this guide is to train yolov3 on darknet and run on pytorch

## Getting Started

These instructions will show you how to run YoloV3 on darknet and pytorch and also provided how to do training on custom object detection


---

## Prerequisites

What things you need to run this codes:

1. Python3
2. Numpy 
3. CUDA -->  [Installation](https://hackmd.io/@ahanjaya/HkioE4SNr) 
4. OpenCV ver. 3. --> [Installation](https://hackmd.io/@ahanjaya/SJaGgKrNr) 

Tested on MSI-GP63 (Leopard 8RE):

1. 8th Gen. Intel® Core™ i7 processor
2. GTX 1060 6GB GDDR5 with desktop level performance
3. Memory 16GB DDR4 2666MHz
4. SSD 256 GB
5. Ubuntu 16.04.06 LTS (with ROS Kinetic)


---
## Table of Contents

[TOC]

---

## Darknet Guide

### How to compile on Linux

The Makefile will attempt to find installed optional dependencies with opencv, openmp, CUDA, cudnn, and build against those. It will also create a shared object library file to use darknet for code development.
```
1. cd ~/Yolo-V3/yolov3_darknet
2. nano Makefile

**Set makefile with following**
GPU=1
CUDNN=1
CUDNN_HALF=0
OPENCV=1
AVX=0
OPENMP=1
LIBSO=0
ZED_CAMERA=0

3. make clean
4. make -j12 #-j12 depends on total of your CPU processor
```


### Download pretrained-weights
```
1. cd ~/Yolo-V3/yolov3_darknet/weights
2. wget https://pjreddie.com/media/files/yolov3.weights
3. wget https://pjreddie.com/media/files/yolov3-tiny.weights
4. wget https://pjreddie.com/media/files/darknet53.conv.74
```

to get yolov3-tiny pretrained weights
```
1. cd ~/Yolo-V3/yolov3_darknet/
2. ./darknet partial cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights yolov3-tiny.conv.15 15
3. mv yolov3-tiny.conv.15 weights/
```


### Detection on a Picture
`./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg -thresh 0.25`


### Real-Time Detection on a Webcam
`./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights /dev/video1 -thresh 0.25`


---

## Pytorch Guide


### Download pretrained-weights
```
1. cd ~/Yolo-V3/yolov3_pytorch/weights
2. sh download_weights.sh
```

### Real-Time Detection on a Webcam
```
1. python3 yolov3_class.py
```


---

## Train on Custom Dataset

### Custom model

Run the commands below to create a custom model definition, replacing <num-classes> with the number of classes in your dataset.

#### YoloV3 Custom model
```
1. cd ~/Yolo-V3/custom_object/
2. bash create_custom_yolov3_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'
```

#### YoloV3-Tiny Custom model
```
1. cd ~/Yolo-V3/custom_object/
2. bash create_custom_yolov3_tiny_model.sh <num-classes> # Will create custom model 'yolov3-tiny-custom.cfg'
```

---

### Classes
Add class names files. This file should have one row per class name.

```
1. cd /Yolo-V3/custom_object/
2. nano custom_classes.names
```


---
### Capturing data
In this example, using guvc-viewer software ubuntu to capture images from webcam.

1. After capturing data, sorting manually by deleting pictures
2. Rename image files --> 

```
1. cd /Yolo-V3/custom_object/
2. python3 rename_images.py
```

---
### Labeling Images (yolo-boobs)

```
1. cd /Yolo-V3/custom_object/
2. cp custom_classes.names custom_classes.txt
3. **double clicks** : boobs.html
4. **click** Images: browse button # select all images in directory
    example:
    select all images in /Yolo-V3/custom_object/custom_object_dataset 
5. **click** Classes: browse button # select all images in directory
    example:
    select /Yolo-V3/custom_object/custom_classes.txt
6. start labeling by using mouse click and next arrow key
7. **click** Save YOLO button # when finish labeling
8. extract label *.txt files into --> /Yolo-V3/custom_object/custom_object_dataset 
```

---

### Spliting Images into Train and Valid

1. cd /Yolo-V3/custom_object/
2. python3 split_train_test.py


---

### Create config for custom data

```
1. cd /Yolo-V3/custom_object/
2. nano custom_classes.data, add following
    1. classes= 1
    2. train  = <path-to-custom>/custom_train.txt
    3. valid  = <path-to-custom>/custom_valid.txt
    4. names  = <path-to-custom>/data/custom_classes.names
    5. backup = <path-to-custom>/backup
```
 
---

### Training

`1. cd /Yolo-V3/yolov3_darknet/`

#### Training from scratch (don`t add weight file in argument)

`./darknet detector train ../custom_object/cfg/custom_classes.data ../custom_object/cfg/yolov3-tiny-custom.cfg `

#### Transfer learning (from pretrained-weights)

`./darknet detector train ../custom_object/cfg/custom_classes.data ../custom_object/cfg/yolov3-tiny-custom.cfg weights/yolov3-tiny.conv.15`

#### Continue learning (from lastest pretrained-weights)
`./darknet detector train ../custom_object/cfg/custom_classes.data ../custom_object/cfg/yolov3-tiny-custom.cfg ../custom_object/backup/yolov3-tiny-custom_last.weights`


After training is complete - get result yolov3-tiny-custom_final.weights from path /Yolo-V3/custom_object/backup/

* every each 100 iterations, weight will backup: Yolo-V3/custom_object/backup/yolov3-tiny-custom_last.weights
* every each 1000 iterations weight will backup: Yolo-V3/custom_object/backup/yolov3-tiny-custom_1000.weights

---

### Testing

#### Test on Darknet

```
1. cd /Yolo-V3/custom_object/backup/
2. cp yolov3-tiny-custom_last.weights ../../yolov3_darknet/weights/

3. cd /Yolo-V3/custom_object/cfg/
4. cp yolov3-tiny-custom.cfg ../../yolov3_darknet/cfg/
5. cp custom_classes.data ../../yolov3_darknet/cfg/

6. cd /Yolo-V3/yolov3_darknet/
7. ./darknet detector demo cfg/custom_classes.data cfg/yolov3-tiny-custom.cfg weights/yolov3-tiny-custom_last.weights /dev/video1 -thresh 0.25

or skip 1-6
8. ./darknet detector demo ../custom_object/cfg/custom_classes.data ../custom_object/cfg/yolov3-tiny-custom.cfg ../custom_object/backup/yolov3-tiny-custom_last.weights` /dev/video1 -thresh 0.25

```


#### Test on Pytorch

```
1. cd /Yolo-V3/custom_object/backup/
2. cp yolov3-tiny-custom_last.weights ../../yolov3_pytorch/weights/

3. cd /Yolo-V3/custom_object/cfg/
4. cp yolov3-tiny-custom.cfg ../../yolov3_pytorch/config/

5. cd /Yolo-V3/custom_object/data/
6. cp custom_classes.names ../../yolov3_pytorch/config/

7. cd /Yolo-V3/yolov3_pytorch/
8. python3 yolov3_custom.py
```



---

References
---
1. https://pjreddie.com/darknet/yolo/
2. https://github.com/cfotache/pytorch_objectdetecttrack
3. https://github.com/drainingsun/boobs

---
## Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::
