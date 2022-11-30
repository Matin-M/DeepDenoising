function output = ApplyMedianFilter(I, wSize)
    output = medfilt2(I, [wSize,wSize]);
end