function output = ApplyNonLocalMeans(I)
    output = imnlmfilt(I);
end