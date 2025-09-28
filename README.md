# 🌊 LWMNet: Lightweight Mamba-centric Wavelet-enhanced Progressive Salient Object Detection in Optical Remote Sensing

Official repository for the paper:  
**"LWMNet: Lightweight Mamba-centric Wavelet-enhanced Progressive Salient Object Detection in Optical Remote Sensing"**

---

## 📖 Overview
Salient Object Detection (SOD) in **Optical Remote Sensing Images (ORSI)** faces challenges such as **complex backgrounds, object scale variation, and limited computational resources**.  
To address these, we propose **LWMNet**, a **lightweight yet effective** framework that integrates:
- **Mamba-centric encoder** for efficient sequence modeling.  
- **Wavelet-enhanced multi-scale feature extraction** to capture both local and global contexts.  
- **Progressive decoding strategy** for precise saliency map generation.  

> 🚀 LWMNet achieves **competitive accuracy with fewer parameters** compared to recent SOTA methods.

---

## 🏗️ Network Architecture
<p align="center">
  <img src="https://github.com/elaxEgan/LWMNet/blob/main/img/LMWNet.jpeg" width="80%">
</p>

---

## 📊 Comparison with State-of-the-Art
<p align="center">
  <img src="https://github.com/elaxEgan/LWMNet/blob/main/img/exp.jpeg" width="90%">
</p>

LWMNet consistently outperforms existing lightweight approaches while maintaining efficiency.

---

## 📦 Resources

🔗 **Pretrained Weights**  
- [Download LWMNet Weights (Baidu)](https://pan.baidu.com/s/1FcqegbFOtavmv4FZLpipEg&pwd=48ow)  

🔗 **Saliency Maps**  
- [Download Saliency Maps (Baidu)](https://pan.baidu.com/s/1E5RBCvGUhaOTOUmxoOVFKA&pwd=shst)  

---

## ⚙️ Installation

Clone this repository:
```bash
git clone https://github.com/elaxEgan/LWMNet.git
cd LWMNet

---

## 🏋️ Training

Download the pre-trained model weights and dataset.
Modify the dataset path in the config file or training script.
Run training:
```bash
python train.py


## 🔎 Inference
Download the trained LWMNet weights.
Set the weight paths in infer.py.
Run inference:
```bash
python infer.py
