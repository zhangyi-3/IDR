# IDR: Self-Supervised Image Denoising via Iterative Data Refinement

[Yi Zhang](https://zhangyi-3.github.io/)<sup>1</sup>,
[Dasong Li]()<sup>1</sup>,
[Ka Lung Law]()<sup>2</sup>,
[Xiaogang Wang](https://scholar.google.com/citations?user=-B5JgjsAAAAJ&hl=zh-CN)<sup>1</sup>,
[Hongwei Qin](https://scholar.google.com/citations?user=ZGM7HfgAAAAJ&hl=en)<sup>2</sup>,
[Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<sup>1</sup><br>
<sup>1</sup>CUHK-SenseTime Joint Lab, <sup>2</sup>SenseTime Research

---

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2111.14358)

This repository is the official PyTorch implementation of [IDR](http://arxiv.org/abs/2111.14358). 
It also includes some personal implementations of well-known unsupervised image denoising methods ([N2N](https://github.com/NVlabs/noise2noise), etc).


### Training
Slurm Training. Find the config name in [configs/synthetic_config.py](configs/synthetic_config.py).
```
sh run_slurm.sh -n config_name

Example of training IDR for Gaussian denoising:
sh run_slurm.sh -n idr-g
```

### SenseNoise dataset
Downloads [Drive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155135732_link_cuhk_edu_hk/ER9Zn20NM5JCs2LtWnJjS88BOnuSOIl69EGvpdUe7t3BIw?e=r0LtAy) | 
[Baidu Netdisk](https://pan.baidu.com/s/1PtqQjGecr24iNwUQ7na1EQ?pwd=05pj)

The released dataset is what we used in our paper. 
Thanks to the advice from the anonymous reviewers, we are still working on improving the quality of the dataset.


 
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
* [N2N](https://github.com/NVlabs/noise2noise), [N2V](https://github.com/juglab/n2v),  [bm3d](https://pypi.org/project/bm3d/), [basicsr](https://github.com/XPixelGroup/BasicSR)
