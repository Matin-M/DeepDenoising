function output = apply_NLM(I)
    output = imnlmfilt(I);
end

function output = bilateral(I, s)
    output = imbilatfilt(I, s);
end