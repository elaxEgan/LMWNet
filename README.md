# LWMNet: Lightweight Mamba-centric Wavelet-enhanced Progressive Salient Object Detection in Optical Remote Sensing

Welcome to the official repository for the paper "LWMNet: Lightweight Mamba-centric Wavelet-enhanced Progressive Salient Object Detection in Optical Remote Sensing".

### Network Architecture

![image](https://github.com/elaxEgan/LWMNet/blob/main/img/LMWNet.jpeg)

### Comparison with SOTA methods

![image](https://github.com/elaxEgan/LWMNet/blob/main/img/exp.jpeg)

### Trained Weights of MTPNet for Testing
We provide Trained Weights of our LMWNet.
[Download](https://pan.baidu.com/s/1FcqegbFOtavmv4FZLpipEg&pwd=48ow)

### Train
Please download the pre-trained model weights and dataset first. Next, generate the path of the training set and the test set, and change the dataset path in the code to the path of the dataset you specified.

~~~python
python train.py
~~~

### Inference
Download the MTPNet model weights, create the necessary directories to store these files, and be sure to update the corresponding paths in the code accordingly. 

~~~python
python infer.py
~~~

### Saliency maps
We provide saliency maps of our LMWNet on ORSSD，EORSSD and ORSI-4199 datasets.
[Download](https://pan.baidu.com/s/1E5RBCvGUhaOTOUmxoOVFKA&pwd=shst)
