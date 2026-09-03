function outImg = preprocessImage(img, targetSize)
% PREPROCESSIMAGE  Resize + light stain/contrast normalization for H&E
% tissue patches.
%
%   outImg = preprocessImage(img, targetSize)
%
%   Steps:
%     1. Ensure 3-channel RGB, uint8
%     2. Resize to targetSize (e.g. [224 224])
%     3. Contrast-limited adaptive histogram equalization (CLAHE) on the
%        luminance channel only, to reduce stain/lighting variability
%        between slides while preserving color (H&E stain hue carries
%        biological meaning, so we don't touch color channels directly).

if size(img,3) == 1
    img = repmat(img, [1 1 3]);
end
img = im2uint8(img);
img = imresize(img, targetSize);

labImg = rgb2lab(img);
L = labImg(:,:,1) / 100;              % normalize L to [0,1] for adapthisteq
L = adapthisteq(L, 'ClipLimit', 0.01);
labImg(:,:,1) = L * 100;

outImg = im2uint8(lab2rgb(labImg));
end
