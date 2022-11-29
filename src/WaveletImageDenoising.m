
% Wavelet Image Denoising 
function output = WaveletImageDenoising(noisyI)
    output = uint8(wdenoise2(noisyI));
end

