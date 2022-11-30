function output = ApplyGaussianFilter(I, hSize, s)
    g = fspecial('gaussian',hSize, s);
    output = conv2(I,g,'same');
end