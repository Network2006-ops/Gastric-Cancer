function buildPipelineFigure(img, className, cfg, outFile)
% BUILDPIPELINEFIGURE  Recreates the 5-column layout from the reference
% figure: Input | Pre-processing | Augmentation | Feature extraction |
% Detection output (with MALIGNANT / SUSPICIOUS / BENIGN labelled boxes)
%
% Detection panel follows the reference image style:
%   - Red   box  =>  MALIGNANT
%   - Yellow box =>  SUSPICIOUS
%   - Blue   box =>  BENIGN
%   with header text: "LABEL | d=X.XX | y=X.XX | R²=X.XX"

pre  = preprocessImage(img, cfg.targetSize);
augs = augmentImage(pre, cfg.numAugPreview);
respMap = featureResponseMap(pre);
[~, ~, detections] = segmentNuclei(pre, className);

%% --- Build detection annotation panel -----------------------------------
detPanel = drawDetectionBoxes(pre, detections);

%% --- Compose figure layout -----------------------------------------------
f  = figure('Visible', 'off', 'Position', [100 100 1400 320]);
tl = tiledlayout(f, 1, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf('Class: %s', className), 'FontWeight', 'bold', 'FontSize', 12);

nexttile; imshow(img);    title('Input image',           'FontSize', 9);
nexttile; imshow(pre);    title('Pre-processing',        'FontSize', 9);

nexttile;
augMontage = imtile(augs, 'GridSize', [2 2]);
imshow(augMontage);       title('Augmentation',          'FontSize', 9);

nexttile;
imagesc(respMap); axis image off; colormap(gca, 'parula');
title('Feature extraction (texture response)', 'FontSize', 9);

nexttile; imshow(detPanel); title('Detection output', 'FontSize', 9);

exportgraphics(f, outFile, 'Resolution', 150);
close(f);
end

%% =========================================================================
function panel = drawDetectionBoxes(img, detections)
% DRAWDETECTIONBOXES  Renders MALIGNANT/SUSPICIOUS/BENIGN bounding boxes
% with colour-coded borders and text labels on an RGB image.
%
%  Box colours:  MALIGNANT => red   [255 0  0  ]
%                SUSPICIOUS=> yellow[255 220 0  ]
%                BENIGN    => blue  [0   120 255]
%
%  Label format (matching reference figure):
%    LABEL | d=X.XX | y=X.XX | R²=X.XX

panel = im2uint8(img);
if isempty(detections)
    return;
end

colorMap = struct( ...
    'MALIGNANT',  uint8([255   0   0]), ...
    'SUSPICIOUS', uint8([255 220   0]), ...
    'BENIGN',     uint8([  0 120 255]));

[H, W, ~] = size(panel);

for k = 1:numel(detections)
    det = detections(k);
    bb  = round(det.bbox);   % [x y w h]

    % Clamp to image bounds
    x1 = max(bb(1), 1);
    y1 = max(bb(2), 1);
    x2 = min(bb(1) + bb(3) - 1, W);
    y2 = min(bb(2) + bb(4) - 1, H);

    if x2 <= x1 || y2 <= y1; continue; end

    c = colorMap.(det.label);   % [R G B]

    % Draw box border (2-pixel wide)
    for t = 0:1
        panel(max(y1-t,1):min(y2+t,H), max(x1-t,1):min(x1+t,W), 1) = c(1);
        panel(max(y1-t,1):min(y2+t,H), max(x1-t,1):min(x1+t,W), 2) = c(2);
        panel(max(y1-t,1):min(y2+t,H), max(x1-t,1):min(x1+t,W), 3) = c(3);

        panel(max(y1-t,1):min(y2+t,H), max(x2-t,1):min(x2+t,W), 1) = c(1);
        panel(max(y1-t,1):min(y2+t,H), max(x2-t,1):min(x2+t,W), 2) = c(2);
        panel(max(y1-t,1):min(y2+t,H), max(x2-t,1):min(x2+t,W), 3) = c(3);

        panel(max(y1-t,1):min(y1+t,H), max(x1-t,1):min(x2+t,W), 1) = c(1);
        panel(max(y1-t,1):min(y1+t,H), max(x1-t,1):min(x2+t,W), 2) = c(2);
        panel(max(y1-t,1):min(y1+t,H), max(x1-t,1):min(x2+t,W), 3) = c(3);

        panel(max(y2-t,1):min(y2+t,H), max(x1-t,1):min(x2+t,W), 1) = c(1);
        panel(max(y2-t,1):min(y2+t,H), max(x1-t,1):min(x2+t,W), 2) = c(2);
        panel(max(y2-t,1):min(y2+t,H), max(x1-t,1):min(x2+t,W), 3) = c(3);
    end
end

% Use insertText from Computer Vision Toolbox if available, else skip text
if license('test', 'Video_and_Image_Blockset')
    positions = zeros(numel(detections), 2);
    labels    = cell(numel(detections), 1);
    colors    = cell(numel(detections), 1);
    for k = 1:numel(detections)
        det  = detections(k);
        bb   = round(det.bbox);
        positions(k,:) = [max(bb(1), 1), max(bb(2)-12, 1)];
        labels{k} = sprintf('%s|d=%.2f|y=%.2f|R^2=%.2f', ...
            det.label, det.conf_d, det.conf_y, det.R2);
        switch det.label
            case 'MALIGNANT';  colors{k} = [255   0   0];
            case 'SUSPICIOUS'; colors{k} = [255 220   0];
            otherwise;         colors{k} = [  0 120 255];
        end
    end
    panel = insertText(panel, positions, labels, ...
        'BoxColor', colors, 'TextColor', 'white', 'FontSize', 7, ...
        'BoxOpacity', 0.75);
end
end
