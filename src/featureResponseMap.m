function respMap = featureResponseMap(img)
% FEATURERESPONSEMAP  Produce a heatmap visualizing local texture
% "activation" across the image -- a classical-CV stand-in for a CNN
% feature/activation map, used only for the figure's visual panel.
%
%   respMap = featureResponseMap(img) returns an HxW double in [0,1],
%   computed as local entropy (texture complexity) combined with
%   gradient magnitude (edge density), both of which spike on
%   cellular/nuclear-dense regions -- visually similar in spirit to
%   what a shallow CNN's early activation maps highlight.
%
%   If you swap in a real trained CNN (see README), replace this
%   function's body with a call to `activations(net, img, layerName)`
%   and take the mean across channels.

grayImg = im2double(rgb2gray(im2uint8(img)));

entropyMap = entropyfilt(grayImg);
entropyMap = mat2gray(entropyMap);

[gx, gy] = imgradientxy(grayImg);
gradMag = mat2gray(sqrt(gx.^2 + gy.^2));

respMap = mat2gray(0.6*entropyMap + 0.4*gradMag);
end
