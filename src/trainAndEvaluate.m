function [model, results] = trainAndEvaluate(X, Y, cfg)
% TRAINANDEVALUATE  Train a multiclass ECOC-SVM on features X/labels Y,
% with a stratified train/test split, and report real (computed, not
% fabricated) accuracy + confusion matrix.

classes = categories(Y);
trainIdx = false(size(Y));
testIdx  = false(size(Y));

for c = 1:numel(classes)
    idx = find(Y == classes{c});
    idx = idx(randperm(numel(idx)));
    nTrain = round(cfg.trainFraction * numel(idx));
    nTrain = max(nTrain, 1);
    nTrain = min(nTrain, numel(idx)-1); % keep at least 1 for test
    trainIdx(idx(1:nTrain)) = true;
    testIdx(idx(nTrain+1:end)) = true;
end

Xtrain = X(trainIdx,:); Ytrain = Y(trainIdx);
Xtest  = X(testIdx,:);  Ytest  = Y(testIdx);

% Standardize features using train-set statistics only
mu = mean(Xtrain, 1);
sigma = std(Xtrain, [], 1); sigma(sigma==0) = 1;
XtrainN = (Xtrain - mu) ./ sigma;
XtestN  = (Xtest  - mu) ./ sigma;

t = templateSVM('KernelFunction', 'linear', 'Standardize', false);
model.classifier = fitcecoc(XtrainN, Ytrain, 'Learners', t);
model.mu = mu;
model.sigma = sigma;

predTest = predict(model.classifier, XtestN);

results.accuracy = mean(predTest == Ytest);
results.confMat = confusionmat(Ytest, predTest);
results.classes = categories(Ytest);
results.Ytest = Ytest;
results.predTest = predTest;
results.trainIdx = trainIdx;
results.testIdx = testIdx;
end
