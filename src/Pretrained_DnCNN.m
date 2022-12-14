I = imread('cameraman.tif');
noisyI = imnoise(I,'gaussian',0,0.01);
figure;montage({I,noisyI})
title('Original Image (Left) and Noisy Image (Right)')

defaultNet = denoisingNetwork('DnCNN');
denoisedI = denoiseImage(noisyI, defaultNet);
figure;imshow(denoisedI)
title('Denoised Image using DnCNN')

%PSNR:
psnr_to_original = psnr(denoisedI, I);
disp("PSNR -> " + psnr_to_original)
%SSIM:
ssim_to_original = ssim(denoisedI, I);
disp("SSIM -> " + ssim_to_original)