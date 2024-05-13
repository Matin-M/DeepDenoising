# DeepDenoising
A performance and quality evaluation of residual learning deep convolutional neural networks for image denoising applications based on the architecture outlined in the paper *IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 26, NO. 7, JULY 2017*

# Brief architecture outline
- 50x50x1 input layer, the input of our DnCNN is a noisy observation modeled by **y = x+v**
- 20 Convolution, ReLU, and normalization layers:
   - 2d convolution w/ 64 3x3x64 convolutions, weights/bias initialized to zeros, and uniform padding across all dimensions, stride set to 1
   - Rectified Linear Unit activation layer (ReLU)
   - Batch normalization layer
- MSE regression output Layer
- Intuition:
  - Theoretically increasing the convolution layers further generalizes features, thus increasing ability to delineate from noise

# Evaluation methodology
- 75 test images, where each image was degraded via Gaussian noise with variance randomly sampled between 0.005 and 0.2
- Objective performance metrics used:
  - SSIM: Correlated with quality/perception of human vision
  - PSNR: Evaluates noise based on mean square error to test image
  - Benchmark filters used as comparison:
    - Median, Gaussian, Wiener, Wavelet, Non-local means, bilateral filter


# MATLAB Dependencies
- Image Processing Toolbox
- Deep Learning Toolbox
- Parallel Compute Toolbox
