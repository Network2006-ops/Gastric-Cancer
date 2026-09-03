function feat = extractFeatures(img)
% EXTRACTFEATURES  Handcrafted color + texture feature vector for a
% tissue patch. Deliberately avoids requiring Deep Learning Toolbox so
% this pipeline runs on a base MATLAB + Image Processing Toolbox +
% Statistics and Machine Learning Toolbox install.
%
%   feat = extractFeatures(img) returns a 1xD row vector:
%     - 16-bin color histogram per RGB channel (48 values)
%     - GLCM texture (Contrast, Correlation, Energy, Homogeneity) at
%       4 offsets on the grayscale image (16 values)
%     - mean/std of each RGB channel (6 values)
%
%   If you have Deep Learning Toolbox + a pretrained network available,
%   see extractDeepFeatures.m for a drop-in replacement that will
%   generally perform much better once you have a real-sized dataset.

img = im2uint8(img);
if size(img,3) == 1
    img = repmat(img, [1 1 3]);
end

% --- Color histograms ---
nBins = 16;
histFeat = [];
for c = 1:3
    h = imhist(img(:,:,c), nBins);
    h = h / sum(h);
    histFeat = [histFeat, h']; %#ok<AGROW>
end

% --- Channel statistics ---
statFeat = [];
for c = 1:3
    channel = double(img(:,:,c));
    statFeat = [statFeat, mean(channel(:)), std(channel(:))]; %#ok<AGROW>
end

% --- GLCM texture ---
grayImg = rgb2gray(img);
offsets = [0 1; -1 1; -1 0; -1 -1];
glcm = graycomatrix(grayImg, 'Offset', offsets, 'Symmetric', true);
stats = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});
textureFeat = [mean(stats.Contrast), mean(stats.Correlation), ...
               mean(stats.Energy), mean(stats.Homogeneity), ...
               stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];

feat = [histFeat, statFeat, textureFeat];
end
