# Semantic-Segmentation-PyTorch
Semantic segmentation models, training, and inference code, implemented in PyTorch

## Supported networks
* [UNet](https://arxiv.org/abs/1505.04597): backbones [MobileNetV2](https://arxiv.org/abs/1801.04381) (all aphas and expansions), [ResNetV1](https://arxiv.org/abs/1512.03385) (all num_layers)
* [DeepLab3+](https://arxiv.org/abs/1802.02611): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18,34,50,101), [VGG16_bn](https://arxiv.org/abs/1409.1556)
* [BiSeNet](https://arxiv.org/abs/1808.00897): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18)
* [PSPNet](https://arxiv.org/abs/1612.01105): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18,34,50,101)
* [ICNet](https://arxiv.org/abs/1704.08545): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18,34,50,101)

To assess architecture, memory, forward time (in either cpu or gpu), numper of parameters, and number of FLOPs of a network, use this command:
```
python measure_model.py
```

## Dataset
**Portrait Segmentation (Human/Background)**
* [Automatic Portrait Segmentation for Image Stylization](http://xiaoyongshen.me/webpage_portrait/index.html): 1800 images
* [Supervisely Person](https://hackernoon.com/releasing-supervisely-person-dataset-for-teaching-machines-to-segment-humans-1f1fc1f28469): 5711 images

## Installation
* Python3.6 is used in this repository.
* To install required packages, use pip:
```
pip install -r requirements.txt
```

## Training
* For training a network from scratch, for example DeepLab3+, use this command:
```
python train.py --config config/config_DeepLab.json --device 0
```
where *config/config_DeepLab.json* is the configuration file which contains network, dataloader, optimizer, losses, metrics, and visualization configurations.
* For resuming training the network from a checkpoint, use this command:
```
python train.py --config config/config_DeepLab.json --device 0 --resume path_to_checkpoint/model_best.pth
```
* One can open tensorboard to monitor the training progress by enabling the visualization mode in the configuration file.

## Inference
There are two modes of inference: [video](https://github.com/AntiAegis/Semantic-Segmentation-PyTorch/blob/master/inference_video.py) and [webcam](https://github.com/AntiAegis/Semantic-Segmentation-PyTorch/blob/master/inference_webcam.py).
```
python inference_video.py --watch --use_cuda --checkpoint path_to_checkpoint/model_best.pth
python inference_webcam.py --use_cuda --checkpoint path_to_checkpoint/model_best.pth
```

## Benchmark
* Networks are trained on a combined dataset from the two mentioned datasets above. There are [6627 training](https://github.com/AntiAegis/Semantic-Segmentation-PyTorch/blob/master/dataset/train_mask.txt) and [737 testing](https://github.com/AntiAegis/Semantic-Segmentation-PyTorch/blob/master/dataset/valid_mask.txt) images.
* Input size of model is set to 320.
* The CPU and GPU time is the averaged inference time of 10 runs (there are also 10 warm-up runs before measuring) with batch size 1.
* The mIoU is measured on the testing subset (737 images) from the combined dataset.
* Hardware configuration for benchmarking:
```
CPU: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
GPU: GeForce GTX 1050 Mobile, CUDA 9.0
```

| Model | Parameters | FLOPs | CPU time | GPU time | mIoU |
|:-:|:-:|:-:|:-:|:-:|:-:|
| UNet_MobileNetV2 (alpha=1.0, expansion=6) | 4.7M | 1.3G | 167ms | 17ms | 91.37% |
| UNet_ResNet18 | 16.6M | 9.1G | 165ms | 21ms | 90.09% |
| DeepLab3+_ResNet18 | 16.6M | 9.1G | 133ms | 28ms | --- |
| BiSeNet_ResNet18 | 11.9M | 4.7G | 88ms | 10ms | --- |
| PSPNet_ResNet18 | 12.6M | 20.7G | 235ms | 666ms | --- |
| ICNet_ResNet18 | 11.6M | 2.0G | 48ms | 55ms | 86.27% |
