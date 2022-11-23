I = imread('cameraman.tif');
noisyI = imnoise(I,'gaussian',0,0.01);
figure;montage({I,noisyI})
title('Original Image (Left) and Noisy Image (Right)')

net = denoisingNetwork('DnCNN');
denoisedI = denoiseImage(noisyI, net);
figure;imshow(denoisedI)
title('Denoised Image using DnCNN')

%PSNR
psnr_to_original = psnr(denoisedI, I);
disp("PSNR -> " + psnr_to_original)
