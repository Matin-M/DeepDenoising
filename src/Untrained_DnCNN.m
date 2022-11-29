MerchDataFileName = 'MerchData.zip';
RandomTrainingImageFileName = 'Unsplash_random_images_collection.zip';
9
MerchData = fullfile(tempdir,'MerchData');
if ~exist(MerchData,'dir')
    unzip(MerchDataFileName,tempdir);
end

RandomImages = fullfile(pwd,'TrainingData','unsplash-images-collection');
if ~exist(RandomImages,'dir')
    unzip(fullfile(pwd,'TrainingData',RandomTrainingImageFileName),fullfile(pwd,'TrainingData'));
end

%Training data store options
imds = imageDatastore(MerchData, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

timds = imageDatastore(RandomImages, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Add zero-mean gaussian white noise w/ stdv = within [x y]
% - PatchesPerImage: 
% - PatchSize: Square if scalar
% - GaussianNoiseLevel: Randomly sample stdv from range for each image
% patch.
dnimds = denoisingImageDatastore(timds,...
    'PatchesPerImage',35,...
    'PatchSize', 50,...
    'GaussianNoiseLevel',[0.005 0.2],...
    'ChannelFormat','grayscale');

minibatch = preview(dnimds);
montage(minibatch.input)
figure
montage(minibatch.response)

%Custom DnCNN layers
customLayers = [
     imageInputLayer([50 50 1])
     convolution2dLayer(3, 64, 'NumChannels', 1, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,1,64), Bias=zeros(1,1,64))
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=0)
     batchNormalizationLayer
     reluLayer
     convolution2dLayer(3, 64, 'NumChannels', 64, 'Stride', [1 1],'Padding', [1 1 1 1], Weights=zeros(3,3,64,64), Bias=zeros(1,1,64), BiasLearnRateFactor=1)
     regressionLayer];

%Training options:
% - solverName: 'sgdm' = Stochastic gradient descent with momentum,
% 'adam' = Adam optimizer
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
    Verbose=true, ...
    DispatchInBackground=true, ...
    VerboseFrequency = 25, ...
    ExecutionEnvironment='gpu');

%Train neural net
[neuralNet_20E_35P, netInfo] = trainNetwork(dnimds, dnCNNLayers, options)