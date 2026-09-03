function augImgs = augmentImage(img, n)
% AUGMENTIMAGE  Generate n randomly augmented versions of img.
%
%   augImgs = augmentImage(img, n) returns a 1xn cell array of uint8
%   images, each a random combination of:
%     - rotation (+/- 25 degrees)
%     - horizontal / vertical flip
%     - brightness jitter (+/- 15%)
%     - mild Gaussian noise (sensor/stain variability proxy)
%
%   This is a lightweight stand-in for MATLAB's imageDataAugmenter,
%   written explicitly so it has zero extra toolbox dependencies beyond
%   Image Processing Toolbox.

augImgs = cell(1, n);
[h, w, ~] = size(img);

for i = 1:n
    out = img;

    angle = (rand()*2 - 1) * 25;
    out = imrotate(out, angle, 'bilinear', 'crop');

    if rand() > 0.5
        out = fliplr(out);
    end
    if rand() > 0.5
        out = flipud(out);
    end

    brightnessFactor = 1 + (rand()*2 - 1) * 0.15;
    out = im2uint8(im2double(out) * brightnessFactor);

    noiseSigma = 0.01 * rand();
    out = im2uint8(imnoise(im2double(out), 'gaussian', 0, noiseSigma^2));

    augImgs{i} = imresize(out, [h w]);
end
end
