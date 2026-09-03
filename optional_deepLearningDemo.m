%% optional_deepLearningDemo.m
% Bonus script: trains an ACTUAL small CNN from scratch on the
% augmented 8-class dataset and shows MATLAB's real training-progress
% plot (true accuracy/loss curves, not simulated).
%
% Requires: Deep Learning Toolbox.
%
% HONESTY NOTE: With only 8 source images (even augmented up to a few
% hundred variants), a CNN trained here will likely overfit hard and
% will NOT reach anything like the paper's 98%+ accuracy. That number
% requires the full 100,000-image NCT-CRC-HE-100K dataset (or similar
% scale) and a proper architecture/training schedule. This script
% exists so you can see a REAL training-progress curve on your own
% machine, and swap in the full dataset later (see README.m).

clear; clc; close all;
addpath(genpath('src'));

if ~license('test','Neural_Network_Toolbox')
    error(['Deep Learning Toolbox not found. This bonus script is optional -- ' ...
           'the main pipeline (main.m) does not need it.']);
end

cfg.rawFolder       = fullfile('data','raw');
cfg.organizedFolder = fullfile('data','organized');
cfg.targetSize       = [224 224];
cfg.numAugPerClass   = 60;
rng(42);

organizeDataset(cfg.rawFolder, cfg.organizedFolder);
imds = imageDatastore(cfg.organizedFolder, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Build an augmented image datastore on disk so trainNetwork can stream it
augFolder = fullfile('data','augmented_for_cnn');
if exist(augFolder,'dir'); rmdir(augFolder,'s'); end
mkdir(augFolder);

for i = 1:numel(imds.Files)
    img = imread(imds.Files{i});
    label = char(imds.Labels(i));
    classFolder = fullfile(augFolder, label);
    if ~exist(classFolder,'dir'); mkdir(classFolder); end

    pre = preprocessImage(img, cfg.targetSize);
    variants = [{pre}, augmentImage(pre, cfg.numAugPerClass - 1)];
    for v = 1:numel(variants)
        imwrite(variants{v}, fullfile(classFolder, sprintf('%s_%03d.png', label, v)));
    end
end

fullImds = imageDatastore(augFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainImds, testImds] = splitEachLabel(fullImds, 0.7, 'randomized');

numClasses = numel(categories(fullImds.Labels));

layers = [
    imageInputLayer([cfg.targetSize 3])

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'ValidationData', testImds, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

fprintf('Training a small CNN from scratch on %d augmented images (%d classes)...\n', ...
    numel(fullImds.Files), numClasses);
net = trainNetwork(trainImds, layers, options);

predTest = classify(net, testImds);
acc = mean(predTest == testImds.Labels);
fprintf('REAL held-out test accuracy on toy dataset: %.2f%%\n', acc*100);
fprintf('(Expect this to be mediocre -- 8 source images is not a real dataset.\n');
fprintf(' Swap in NCT-CRC-HE-100K for meaningful numbers -- see README.md.)\n');

save(fullfile('output','cnn_demo_net.mat'), 'net');
