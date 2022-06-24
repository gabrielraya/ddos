# Diffusing and denoising samples for out-of-distribution detection



This repository contains the official implementation of our paper [Diffusing and denoising samples for out-of-distribution detection](https://openreview.net/forum?id=eATJn1VZQia)


We propose Diffusing and Denoising OOD Samples (DDOS), a new unsupervised method for out-of-distribution detection
that consists of partially diffusing and denoising data with score-based diffusion generative models (SDMs).
In particular, we show that we can successfully reconstruct partially diffused in-distribution samples, indicating a low reconstruction error.
Contrary, OOD samples are pushed far from their initial state resulting in a high reconstruction error. This reconstruction error is used to
distinguish in- from OOD samples, yielding an AUROC of 0.98 on FashionMNIST vs. MNIST.


The basic idea is captured in the figure below:

![schematic](./images/results.PNG)


We first trained an SDM on FashionMNIST with a forward diffusing process defined by the
Itô stochastic differential equation (SDE) $dx = \sigma^t dW$. Then w e perform partial diffusion using the corresponding transition probability

$$
p(x_t|x_0) = \mathcal{N}\left(x_t; x_0, \frac{1}{2\ln \sigma}(\sigma^{2t}-1) I\right)
$$


This transition probability maps an initial or clean state, first column (labeled INPUTS), to a partially diffused state 
, second column (DIFFUSING). The third column (DENOISED) shows the denoised samples obtained by using the the reverse-SDE
in a SDM. _Notice that the denoised IND samples are visually similar to the clean ones_. In contrast, _the denoised OOD 
samples are visually different from the original ones_; they seem to be mapped to a different region in the data 
space, i.e., to the "clothing" class. The rightmost part of the figure shows two $6 \times 6$ heatmaps for the  
reconstruction errors for the IND (top part) and OOD (bottom part) samples. The more saturated the blue color, the higher the error. For both the IND and OOD samples, the associated $6 \times 6$ pixel colors represent the reconstruction errors for the $6 \times 6$ input images on the left. The reconstruction errors for the IND samples are smaller than those for the OOD samples as is evident by comparing the level of saturation of the colors. The errors associated with the OOD samples are darker than those for the IND samples. The OOD detection AUC is equal to $0.98$.


## Notebook

In [Tutorial_on_out_out_distribution_detection_with_SGM.ipynb](./Tutorial_on_out_out_distribution_detection_with_SGM.ipynb) 
we provide a short tutorial. 

## Running experiments

