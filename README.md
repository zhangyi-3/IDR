[Self-Supervised Image Denoising via Iterative Data Refinement](http://arxiv.org/abs/2111.14358)
---
[Yi Zhang](https://zhangyi-3.github.io/)<sup>1</sup>,
[Dasong Li]()<sup>1</sup>,
[Ka Lung Law]()<sup>2</sup>,
[Xiaogang Wang](https://scholar.google.com/citations?user=-B5JgjsAAAAJ&hl=zh-CN)<sup>1</sup>,
[Hongwei Qin](https://scholar.google.com/citations?user=ZGM7HfgAAAAJ&hl=en)<sup>2</sup>,
[Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<sup>1</sup><br>

<sup>1</sup>CUHK-SenseTime Joint Lab, <sup>2</sup>SenseTime Research



### Abstract

> The lack of large-scale noisy-clean image pairs restricts the supervised denoising methods' deployment
in actual applications. While existing unsupervised methods are able to learn image denoising without 
ground-truth clean images, they either show poor performance or work under impractical settings 
(e.g., paired noisy images). In this paper, we present a practical unsupervised image denoising method 
to achieve state-of-the-art denoising performance. Our method only requires single noisy images and a 
noise model, which is easily accessible in practical raw image denoising. It performs two steps 
iteratively: (1) Constructing noisier-noisy dataset with random noise from the noise model; 
(2) training a model on the noisier-noisy dataset and using the trained model to refine noisy images 
as the targets used in the next round. We further approximate our full iterative method with a fast 
algorithm for more efficient training while keeping its original high performance. Experiments on 
real-world noise, synthetic noise, and correlated noise show that our proposed unsupervised denoising 
approach has superior performances to existing unsupervised methods and competitive performance with 
supervised methods. In addition, we argue that existing denoising datasets are of low quality and 
contain only a small number of scenes. To evaluate raw images denoising performance in real-world 
applications, we build a high-quality raw image dataset SenseNoise-500 that contains 
500 real-life scenes. The dataset can serve as a strong benchmark for better evaluating raw image 
denoising.

### Code & Dataset

coming soon ！