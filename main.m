%% main.m
% Gastric Cancer Histopathology Tissue Analysis Pipeline
% -------------------------------------------------------------
% Reproduces, on your 8-class sample dataset, the pipeline shown in
% your reference figure:
%   Input image -> Pre-processing -> Augmentation -> Feature extraction
%   -> Detection/segmentation output
%
% Classes (NCT-CRC-HE-100K convention):
%   ADI = Adipose, DEB = Debris, LYM = Lymphocytes, MUC = Mucus,
%   MUS = Muscle, NOR = Normal mucosa, STR = Stroma, TUM = Tumor
%
% HONESTY NOTE:
%   You gave me ONE image per class (8 images total). That is not
%   enough data to train a real deep network or reproduce a paper's
%   98%+ accuracy / 50-epoch training curves. This script instead:
%     1. Builds a genuinely runnable pipeline on your images
%     2. Uses augmentation to synthesize a small-but-real training set
%     3. Trains an actual multi-class SVM and reports REAL (not
%        fabricated) accuracy/confusion results on that toy set
%     4. Produces an honestly-computed "accuracy vs. training-set-size"
%        curve (analogous in spirit to an accuracy-vs-epoch curve)
%   See README.md section "Extending to the real dataset" for how to
%   swap in the full NCT-CRC-HE-100K dataset (100,000 images) to get
%   results that are actually comparable to published papers.
% -------------------------------------------------------------

clear; clc; close all;
addpath(genpath('src'));

%% ---- Config ----
cfg.rawFolder        = fullfile('data','raw');
cfg.organizedFolder  = fullfile('data','organized');
cfg.outputFolder     = fullfile('output');
cfg.targetSize        = [224 224];
cfg.numAugPerClass    = 40;   % synthetic samples generated per class
cfg.numAugPreview     = 4;    % how many augmented previews to show in the grid figure
cfg.trainFraction     = 0.7;
cfg.rngSeed           = 42;

if ~exist(cfg.outputFolder,'dir'); mkdir(cfg.outputFolder); end
rng(cfg.rngSeed);

fprintf('=== Step 1/5: Organizing dataset by class ===\n');
classNames = organizeDataset(cfg.rawFolder, cfg.organizedFolder);
fprintf('Found classes: %s\n', strjoin(classNames, ', '));

fprintf('\n=== Step 2/5: Building per-image pipeline figures ===\n');
imds = imageDatastore(cfg.organizedFolder, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'FileExtensions', {'.png','.jpg','.jpeg'});

for i = 1:numel(imds.Files)
    img = imread(imds.Files{i});
    className = char(imds.Labels(i));
    outFile = fullfile(cfg.outputFolder, sprintf('pipeline_%s.png', className));
    buildPipelineFigure(img, className, cfg, outFile);
    fprintf('  Saved %s\n', outFile);
end

fprintf('\n=== Step 3/5: Synthesizing augmented training data + extracting features ===\n');
[X, Y] = buildFeatureDataset(imds, cfg);
fprintf('  Built feature matrix: %d samples x %d features across %d classes\n', ...
    size(X,1), size(X,2), numel(unique(Y)));

fprintf('\n=== Step 4/5: Training classifier and evaluating (REAL, computed results) ===\n');
[model, results] = trainAndEvaluate(X, Y, cfg);
fprintf('  Held-out test accuracy: %.2f%%\n', results.accuracy*100);

fprintf('\n=== Step 5/5: Plotting confusion matrix + learning curve ===\n');
plotConfusionAndCurve(X, Y, results, cfg);

fprintf('\n=== Step 6/6: Plotting separate paper figures ===\n');
plotPaperFigures(cfg.outputFolder);

fprintf('\nAll done. Check the "output" folder for figures.\n');
