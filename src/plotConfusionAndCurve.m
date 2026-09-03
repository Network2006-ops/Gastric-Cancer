function plotConfusionAndCurve(X, Y, results, cfg)
% PLOTCONFUSIONANDCURVE  Saves a confusion matrix figure, and a REAL
% (computed, not simulated) accuracy-vs-training-set-size learning
% curve, as the honest analogue of the paper's accuracy-vs-epoch plot.
% We don't do epoch-based deep training here (see README for why), so
% instead we sweep how many augmented training samples per class the
% SVM gets, and measure held-out accuracy at each size -- a real,
% reproducible curve computed on your data.

% --- Confusion matrix ---
f1 = figure('Visible', 'off', 'Position', [100 100 500 450]);
confusionchart(results.confMat, results.classes, ...
    'Title', sprintf('Confusion Matrix (test accuracy = %.1f%%)', results.accuracy*100), ...
    'RowSummary', 'row-normalized');
exportgraphics(f1, fullfile(cfg.outputFolder, 'confusion_matrix.png'), 'Resolution', 150);
close(f1);

% --- Accuracy vs. training-set-size curve (real, computed) ---
classes = categories(Y);
sizesToTry = round(linspace(2, cfg.numAugPerClass - 5, 6));
accByTrainSize = zeros(size(sizesToTry));

for s = 1:numel(sizesToTry)
    n = sizesToTry(s);
    trainIdx = false(size(Y));
    testIdx  = false(size(Y));
    for c = 1:numel(classes)
        idx = find(Y == classes{c});
        idx = idx(randperm(numel(idx)));
        nTrain = min(n, numel(idx)-1);
        trainIdx(idx(1:nTrain)) = true;
        testIdx(idx(nTrain+1:end)) = true;
    end

    Xtr = X(trainIdx,:); Ytr = Y(trainIdx);
    Xte = X(testIdx,:);  Yte = Y(testIdx);

    mu = mean(Xtr,1); sigma = std(Xtr,[],1); sigma(sigma==0) = 1;
    XtrN = (Xtr - mu) ./ sigma;
    XteN = (Xte - mu) ./ sigma;

    t = templateSVM('KernelFunction', 'linear', 'Standardize', false);
    mdl = fitcecoc(XtrN, Ytr, 'Learners', t);
    pred = predict(mdl, XteN);
    accByTrainSize(s) = mean(pred == Yte);
end

f2 = figure('Visible', 'off', 'Position', [100 100 500 400]);
plot(sizesToTry, accByTrainSize*100, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'auto');
grid on;
xlabel('Training samples per class (augmented)');
ylabel('Held-out accuracy (%)');
title({'Accuracy vs. training-set size', '(real, computed -- NOT epoch-based deep training)'});
ylim([0 100]);
exportgraphics(f2, fullfile(cfg.outputFolder, 'learning_curve.png'), 'Resolution', 150);
close(f2);
end
