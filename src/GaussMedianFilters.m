% Median Filtering
function output = apply_median(I, wSize)
    output = medfilt2(I, [wSize,wSize]);
end

% Gaussian Filtering
function output = apply_gaussian(I, hSize, s)
    g = fspecial('gaussian',hSize, s);
    output = conv2(I,g,'same');
end