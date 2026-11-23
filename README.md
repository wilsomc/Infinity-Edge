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

# Setup
1. Create a virtual environment
Type `python -m venv venv` in command line
2. Install dependencies
Type `pip install -r requirements.txt` in command line
3. Download the dataset and put it in `data/` or `dataloader/`
The dataset for this repository can be found [here](https://drive.google.com/drive/folders/11aBwpUVXk4ZO7JKhR9zIva8kUvEdaV2Q?usp=sharing)
4. Run the model
5. Check `results/` for the output

