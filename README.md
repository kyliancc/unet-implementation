# unet-implementation
My U-Net implementation using PyTorch.

**This PyTorch implementation is preparation for my DDPM (Deep Diffusion Probablistic Model) implementation. Waiting for that!** 😁

U-Net original research paper: https://arxiv.org/pdf/1505.04597

> **WARN:**  
This U-Net implementation is modified for the equal size between input and output. 
So all non-padding 3x3 convolution layers was changed with padding=1.
Besides, the copy and crop trick was changed to simply concatenation.
