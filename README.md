# Infinity Edge

Utilizes U-Net encoder and ResNET decoder to predict
depth maps of RBG images.

Our encoder's feature maps
| Name | Channels | Resolution |
| --- | --- | --- |
| skip1 (after conv1/bn/relu) | 64 | H/2 |
| skip2 (layer1) | 64 | H/4 |
| skip3 (layer2) | 128 | H/8 |
| skip4 (layer3) | 256 | H/16 |
| bottleneck (layer4) | 512 | H/32 |
