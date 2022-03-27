[IDR: Self-Supervised Image Denoising via Iterative Data Refinement](http://arxiv.org/abs/2111.14358)
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

### Testing
The code has been tested with the following environment:
```
pytorch == 1.5.0
bm3d == 3.0.7
scipy == 1.4.1 
```
    
- Prepare the datasets. (kodak | BSDS300 | BSD68)
- Download the [pretrained models](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155135732_link_cuhk_edu_hk/Ep0gRwX0hIFKvOSiq5x1QbsBfSmGma1CNxQ8LeMiE93wEw?e=dJxcx3)
 and put them into the checkpoint folder.
- Modify the data root path and noise type (gaussian | gaussian_gray | line | binomial | impulse | pattern).
```
python -u test.py --root your_data_root --ntype gaussian 
```

### SenseNoise dataset
[Download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155135732_link_cuhk_edu_hk/ER9Zn20NM5JCs2LtWnJjS88BOnuSOIl69EGvpdUe7t3BIw?e=r0LtAy)

The released dataset is what we used in our paper. 
Thanks to the advice from the anonymous reviewers, we are still working on improving the quality of the dataset.


### Training code

coming soon ！


### Citation
``` bibtex
@inproceedings{zhang2021IDR,
      title={IDR: Self-Supervised Image Denoising via Iterative Data Refinement},
      author={Zhang, Yi and Li, Dasong and Law, Ka Lung and Wang, Xiaogang and Qin, Hongwei and Li, Hongsheng},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2022}
}
```

### Contact
Feel free to contact zhangyi@link.cuhk.edu.hk if you have any questions.

### Acknowledgments
* [N2N](https://github.com/NVlabs/noise2noise)
* [N2V](https://github.com/juglab/n2v)
* [bm3d](https://pypi.org/project/bm3d/)