# Human-Segmentation-PyTorch
Human segmentation using Deep Learning, implemented in PyTorch

## Supported networks
* [UNet](https://arxiv.org/abs/1505.04597): backbones [MobileNetV2](https://arxiv.org/abs/1801.04381) (all aphas and expansions), [ResNetV1](https://arxiv.org/abs/1512.03385) (all num_layers)
* [DeepLab3+](https://arxiv.org/abs/1802.02611): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18,34,50,101), [VGG16_bn](https://arxiv.org/abs/1409.1556)
* [BiSeNet](https://arxiv.org/abs/1808.00897): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18)
* [PSPNet](https://arxiv.org/abs/1612.01105): backbones [ResNetV1](https://arxiv.org/abs/1512.03385) (num_layers=18,34,50,101)

## Dataset:
* [Automatic Portrait Segmentation for Image Stylization](http://xiaoyongshen.me/webpage_portrait/index.html): 1800 images
* [Supervisely Person](https://hackernoon.com/releasing-supervisely-person-dataset-for-teaching-machines-to-segment-humans-1f1fc1f28469): 5711 images
