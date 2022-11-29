% Wavelet Image Denoising 

img = im2double(rgb2gray(imread('HW04_input.png')));
noised_image = imnoise(blurredImg,'gaussian',0,powerOfSig/1000);
denoise_image = wdenoise2(img);

figure; imshow(img);title("Original Image");

subplot(1,2,1)
imagesc(noised_image)
title('Noised Image')
subplot(1,2,2)
imagesc(denoise_image)
title('Denoised Image')
colormap gray