# FPM-GAN
FPM-GAN is a craniofacial reconstruction model that learns skull–face correspondence using a frequency-perception generator and multi-scale discriminator. With semi-wavelet attention, high–low frequency decomposition, and adaptive loss, it improves global structure and local texture restoration.

## Setup

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

- ### Getting Started
- Clone this repo:
```bash
git clone git@github.com:MaLongqian/FPM-GAN.git 
cd FPM-GAN

- Train the model
```bash
python train.py --dataroot ./datasets/facades --name facades_FPM_GAN --model FPM_GAN --direction BtoA
python -m visdom.server
```
- Test the model:
```bash
python test.py --dataroot ./datasets/facades --name facades_FPM_GAN--model FPM_GAN --direction BtoA
