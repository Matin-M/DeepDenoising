filename = 'MerchData.zip';

MerchData = fullfile(tempdir,'MerchData');
if ~exist(MerchData,'dir')
    unzip(filename,tempdir);
end

%Testing image data store
imds = imageDatastore(MerchData, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

clear PSNR_dict;
clear SSIM_dict;
PSNR_dict = dictionary();
SSIM_dict = dictionary();

%Initialize dictionaries
PSNR_dict("______Average PSNR______") = 1;
SSIM_dict("______Average SSIM______") = 1;

%Define the number of test images to use
numIterations = input("Please specify the number of images to evaluate: ");
stdvRange = 0.006+rand(1,numIterations)*(0.19-0.006);

%Evaluate perf of denosing methods
for k=1:numIterations
    I=rgb2gray(readimage(imds,k));
    noisyI = imnoise(I,'gaussian',0,stdvRange(k));
    %Evaluate ML networks
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, defaultNet, "Default");
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, neuralNet_20E_25P, "neuralNet_20E_25P");
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, neuralNet_20E_30P, "neuralNet_20E_30P");
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, neuralNet_20E_35P, "neuralNet_20E_35P");
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, neuralNet_20E_40P, "neuralNet_20E_40P");
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, neuralNet_20E_45P, "neuralNet_20E_45P");
    %Evaluate other denoising methods
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, @medfilt2, "Median_Filt");
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, @imgaussfilt, "Gaussian_Filt");
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, @WienerFilt, "Wiener_Filt");
    [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict,I, noisyI, @WaveletImageDenoising, "Wavelet_Filt");
end

%Compute averages
PSNR_dict(keys(PSNR_dict)) = PSNR_dict(keys(PSNR_dict))/numIterations;
SSIM_dict(keys(SSIM_dict)) = SSIM_dict(keys(SSIM_dict))/numIterations;

disp(PSNR_dict)
disp(SSIM_dict)

%The denoiser parameter is the denosing method. It can either be a filtering function or
%a multilayer DnCNN
function [PSNR_dict, SSIM_dict] = recordVals(PSNR_dict, SSIM_dict, I, noisyI, denoiser, Key)
    if(~isKey(PSNR_dict, Key))
        PSNR_dict(Key) = 0;
    end
    if(~isKey(SSIM_dict, Key))
        SSIM_dict(Key) = 0;
    end
    %denoiseImage only works with neural nets, use a different function here to
    %denoise an image via other approaches like gaussian, median filtering,
    %laplacian, etc. 
    %Might need to add some kind of check here that determines if the
    %denoisier parameter is a neural net or a traditional denosing function
    if(isa(denoiser,'SeriesNetwork'))
        denoisedI = denoiseImage(noisyI, denoiser);
    else
        denoisedI = denoiser(noisyI);
    end
    PSNR_dict(Key) = PSNR_dict(Key) + psnr(denoisedI, I);
    SSIM_dict(Key) = SSIM_dict(Key) + ssim(denoisedI, I);
end
