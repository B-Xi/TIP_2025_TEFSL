# TEFSL
This repo is the implementation of the following paper:

**Transductive Few-shot Learning With Enhanced Spectral-Spatial Embedding for Hyperspectral Image Classification** (TIP 2025), [[paper]](DOI:10.1109/TIP.2025.3531709)

## Abstract
Few-shot learning (FSL) has been rapidly developed in the hyperspectral image (HSI) classification, potentially eliminating time-consuming and costly labeled data acquisition requirements. Effective feature embedding is empirically significant in FSL methods, which is still challenging for the HSI with rich spectral-spatial information. In addition, compared with inductive FSL, transductive models typically perform better as they explicitly leverage the statistics in the query set. To this end, we devise a transductive FSL framework with enhanced spectral-spatial embedding (TEFSL) to fully exploit the limited prior information available. First, to improve the informative features and suppress the redundant ones contained in the HSI, we devise an attentive feature embedding network (AFEN) comprising a channel calibration module (CCM). Next, a meta-feature interaction module (MFIM) is designed to optimize the support and query features by learning adaptive co-attention using convolutional filters. During inference, we propose an iterative graph-based prototype refinement scheme (iGPRS) to achieve test-time adaptation, making the class centers more representative in a transductive learning manner. Extensive experimental results on four standard benchmarks demonstrate the superiority of our model with various handfuls (i.e., from 1 to 5) labeled samples. 
The code will be available online at https://github.com/B-Xi/TIP_2025_TEFSL.

## Training and Test Process
1. Prepare the training and test data as operated in the paper.
2. Run the 'TEFSL_UP.py' to reproduce the TEFSL results on UP data set.