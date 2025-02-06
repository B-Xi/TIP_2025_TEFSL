# TEFSL
This repo is the implementation of the following paper:

Transductive Few-shot Learning With Enhanced Spectral-Spatial Embedding for Hyperspectral Image Classification, TIP, 2025.
==
[Bobo Xi](https://b-xi.github.io/), [Yun Zhang](https://ieeexplore.ieee.org/author/37087032130), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yan Huang](https://scholar.google.com/citations?user=SgVl7O0AAAAJ&hl=zh-CN), [Zan Li](https://scholar.google.com/citations?hl=zh-CN&user=FL3Mj4MAAAAJ&view_op=list_works&sortby=pubdate), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), and [Jocelyn Chanussot](https://jocelyn-chanussot.net/).
***

Code for the paper: [Transductive Few-shot Learning With Enhanced Spectral-Spatial Embedding for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/10855324).

<div align=center><img src="/TEFSL.png" width="90%" height="90%"></div>
Fig. 1: The architecture of the proposed TEFSL for few-shot HSIC. 

## Abstract
Few-shot learning (FSL) has been rapidly developed in the hyperspectral image (HSI) classification, potentially eliminating time-consuming and costly labeled data acquisition requirements. Effective feature embedding is empirically significant in FSL methods, which is still challenging for the HSI with rich spectral-spatial information. In addition, compared with inductive FSL, transductive models typically perform better as they explicitly leverage the statistics in the query set. To this end, we devise a transductive FSL framework with enhanced spectral-spatial embedding (TEFSL) to fully exploit the limited prior information available. First, to improve the informative features and suppress the redundant ones contained in the HSI, we devise an attentive feature embedding network (AFEN) comprising a channel calibration module (CCM). Next, a meta-feature interaction module (MFIM) is designed to optimize the support and query features by learning adaptive co-attention using convolutional filters. During inference, we propose an iterative graph-based prototype refinement scheme (iGPRS) to achieve test-time adaptation, making the class centers more representative in a transductive learning manner. Extensive experimental results on four standard benchmarks demonstrate the superiority of our model with various handfuls (i.e., from 1 to 5) labeled samples. 
The code will be available online at https://github.com/B-Xi/TIP_2025_TEFSL.

## Training and Test Process
1. Prepare the training and test data as operated in the paper.
2. Run the 'TEFSL_UP.py' to reproduce the TEFSL results on UP data set.

## References
--
If you find this code helpful, please kindly cite:

[1] Xi, B., Zhang, Y., Li, J., Huang, Y., Li, Y., Li, Z., & Chanussot, J, "Transductive Few-Shot Learning With Enhanced Spectral-Spatial Embedding for Hyperspectral Image Classification," in IEEE Transactions on Image Processing, vol. 34, pp. 854-868, 2025, doi: 10.1109/TIP.2025.3531709.

Citation Details
--
BibTeX entry:
```
@ARTICLE{TIP_2025_TEFSL,
  author={Xi, Bobo and Zhang, Yun and Li, Jiaojiao and Huang, Yan and Li, Yunsong and Li, Zan and Chanussot, Jocelyn},
  journal={IEEE Transactions on Image Processing}, 
  title={Transductive Few-Shot Learning With Enhanced Spectral-Spatial Embedding for Hyperspectral Image Classification}, 
  year={2025},
  volume={34},
  number={},
  pages={854-868},
  doi={10.1109/TIP.2025.3531709}}
```
 
Licensing
--
Copyright (C) 2025 Bobo Xi, Yun Zhang

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
