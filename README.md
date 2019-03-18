# Human-Segmentation-PyTorch
Human segmentation using Deep Learning, implemented in PyTorch

## Supported networks
* [UNet](https://arxiv.org/abs/1505.04597): backbones [MobileNetV2](https://arxiv.org/abs/1801.04381) (all aphas and expansions), [ResNetV1](https://arxiv.org/abs/1512.03385) (all num_layers)
* [DeepLab3+](https://arxiv.org/abs/1802.02611): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18,34,50,101), [VGG16_bn](https://arxiv.org/abs/1409.1556)
* [BiSeNet](https://arxiv.org/abs/1808.00897): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18)
* [PSPNet](https://arxiv.org/abs/1612.01105): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18,34,50,101)

To assess architecture, memory, forward time (in either cpu or gpu), numper of parameters, and number of FLOPs of a network, use this command:
```
python measure_model.py
```

## Dataset
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
