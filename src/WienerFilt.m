function denoised = WienerFilt(noisyI)
    denoised = wiener2(noisyI,[5 5])
end

