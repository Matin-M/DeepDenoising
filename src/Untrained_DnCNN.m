url = 'http://download.tensorflow.org/example_images/flower_photos.tgz';
downloadFolder = tempdir;
filename = fullfile(downloadFolder,'flower_dataset.tgz');

dataFolder = fullfile(downloadFolder,'flower_photos');
if ~exist(dataFolder,'dir')
    fprintf("Downloading Flowers data set (218 MB)... ")
    websave(filename,url);
    untar(filename,downloadFolder)
    fprintf("Done.\n")
end

imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Add zero-mean gaussian white noise w/ stdv = within [x y]
% - PatchesPerImage: 
% - PatchSize: Square if scalar
% - GaussianNoiseLevel: Randomly sample stdv from range for each image
% patch.
dnimds = denoisingImageDatastore(imds,...
    'PatchesPerImage',512,...
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
options = trainingOptions("sgdm", ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=5, ...
    MaxEpochs=20, ...
    MiniBatchSize=512, ...
    Verbose=true,...
    Plots="training-progress", ...
    ExecutionEnvironment='auto');

[net, info] = trainNetwork(dnimds, layers, options)