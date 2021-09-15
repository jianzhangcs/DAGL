# Dynamic Attentive Graph Learning for Image Restoration
This repository is for GATIR introduced in the following paper:  
Chong Mou, [Jian Zhang](https://jianzhang.tech/), Zhuoyuan Wu; Dynamic Attentive Graph Learning for Image Restoration; IEEE International Conference on Computer Vision (ICCV) 2021  
[\[arxiv\]](https://arxiv.org/abs/2109.06620)  
Code is coming soon!  
## Requirements
- Python 3.6
- PyTorch >= 1.1.0
- numpy
- skimage
- cv2  
## Introduction  
In this paper, we propose an improved graph attention model for image restoration. Unlike previous non-local image restoration methods, our model can assign an adaptive number of neighbors for each query item and construct long-range correlations based on feature patches. Furthermore, our proposed dynamic attentive graph learning can be easily extended to other computer vision tasks. Extensive experiments demonstrate that our proposed model achieves state-of-the-art performance on wide image restoration tasks: synthetic image denoising, real image denoising, image demosaicing, and compression artifact reduction.  

![Network](/Figs/graph.PNG)
## Citation
If you find our work helpful in your resarch or work, please cite the following papers.
```
@article{mou2021gatir,
  title={Dynamic Attentive Graph Learning for Image Restoration},
  author={Chong, Mou and Jian, Zhang and Zhuoyuan, Wu},
  journal={IEEE International Conference on Computer Vision},
  year={2021}
}
```


