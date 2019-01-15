# Noise-Level-Estimation

This is the reference implementation of [Single-Image Noise Level Estimation for Blind Denoising](https://ieeexplore.ieee.org/document/6607209). In this paper, a patch-based noise level estimation algorithm is presented. Approach includes the process of selecting low-rank patches without high frequency components from a single noisy image. The selection is based on the gradients of the patches and their statistics. Then, the noise level is estimated from the selected patches using principal component analysis.
