function [X, Y] = buildFeatureDataset(imds, cfg)
% BUILDFEATUREDATASET  For each source image, generate
% cfg.numAugPerClass augmented variants and extract features from each,
% producing a feature matrix X (rows = samples) and label vector Y.
%
%   CAVEAT (read this before trusting downstream accuracy numbers):
%   All augmented samples for a class are derived from a SINGLE source
%   image. This means train/test splits below are optimistic -- the
%   model can partly learn "this exact slide's noise pattern" rather
%   than general tissue-class structure. With only 8 source images
%   total, there is no way around this. Real evaluation requires
%   multiple independent patient images per class (see README).

X = [];
Y = categorical.empty(0,1);

for i = 1:numel(imds.Files)
    img = imread(imds.Files{i});
    label = imds.Labels(i);

    pre = preprocessImage(img, cfg.targetSize);
    variants = [{pre}, augmentImage(pre, cfg.numAugPerClass - 1)];

    for v = 1:numel(variants)
        f = extractFeatures(variants{v});
        X = [X; f]; %#ok<AGROW>
        Y = [Y; label]; %#ok<AGROW>
    end
end
end
