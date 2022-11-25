filename = 'MerchData.zip';

MerchData = fullfile(tempdir,'MerchData');
if ~exist(MerchData,'dir')
    unzip(filename,tempdir);
end

url = 'http://download.tensorflow.org/example_images/flower_photos.tgz';
downloadFolder = tempdir;
filename = fullfile(downloadFolder,'flower_dataset.tgz');

FlowerData = fullfile(downloadFolder,'flower_photos');
if ~exist(FlowerData,'dir')
    fprintf("Downloading Flowers data set (218 MB)... ")
    websave(filename,url);
    untar(filename,downloadFolder)
    fprintf("Done.\n")
end

imds = imageDatastore(MerchData, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Add zero-mean gaussian white noise w/ stdv = within [x y]
% - PatchesPerImage: 
% - PatchSize: Square if scalar
% - GaussianNoiseLevel: Randomly sample stdv from range for each image
% patch.
dnimds = denoisingImageDatastore(imds,...
    'PatchesPerImage',50,...
    'PatchSize', 50,...
    'GaussianNoiseLevel',[0.01 0.1],...
    'ChannelFormat','grayscale');

minibatch = preview(dnimds);
montage(minibatch.input)
figure
montage(minibatch.response)

%Predefined denoising layers
layers = dnCNNLayers;

%Training options:
% - solverName: 'sgdm' = Stochastic gradient descent with momentum
% optimizer, 'rmsprop' = RMSProp optimizer, 'adam' = Adam optimizer
% - LearnRateSchedule: The algorithm updates the learning rate by LearnRateDropFactor after a
% certain number of epochs, specified by LearnRateDropPeriod
% - MaxEpochs: The total number of passes performed during training
options = trainingOptions("adam", ...
    InitialLearnRate=3e-4, ...
    SquaredGradientDecayFactor=0.99, ...
    MaxEpochs=20, ...
    MiniBatchSize=128, ...
    Plots="training-progress", ...
    ExecutionEnvironment='gpu');

[net, info] = trainNetwork(dnimds, layers, options)

%Generate noisy image
I = imread('cameraman.tif');
noisyI = imnoise(I,'gaussian',0,0.01);
montage({I,noisyI})
title('Original Image (Left) and Noisy Image (Right)')

%Denoise noisy image using DnCNN
denoisedI = denoiseImage(noisyI,net);
imshow(denoisedI)
title('Denoised Image')

%Performance Metrics
%PSNR:
psnr_to_original = psnr(denoisedI, I);
disp("PSNR -> " + psnr_to_original)
%SSIM:
ssim_to_original = ssim(denoisedI, I);
disp("SSIM -> " + ssim_to_original)