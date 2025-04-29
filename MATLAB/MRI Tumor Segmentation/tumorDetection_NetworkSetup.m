% MRI Tumor detection by regionprops and network training 
clear all 


%Load dataset 
datasetPath = uigetdir(pwd, 'Select Dataset Folder');
imageFiles = dir(fullfile(datasetPath, '*.png')); % or '*.jpg', '*.bmp'
%
numImages = length(imageFiles);
features = [];
labels = [];

for i = 1:numImages
    % Load image
    img = imread(fullfile(datasetPath, imageFiles(i).name));
    img = rgb2gray(img);


    % Preprocessing
    img = medfilt2(img, [3 3]); % Denoising
    bw = imbinarize(img, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.5);
    bw = imopen(bw, strel('disk',3));
    bw = imfill(bw, 'holes');

    %Identify region properties geometrical 
    stats = regionprops(bw, 'Area', 'Perimeter', 'Eccentricity', 'Solidity', 'Extent', 'MajorAxisLength', 'MinorAxisLength');

    if isempty(stats)
        % No tumor detected
        featureVector = zeros(1,7); % 7 properties
    else
        % Assume the largest region is the tumor, can change by threshold 
        [~, idx] = max([stats.Area]);
        s = stats(idx);
        featureVector = [s.Area, s.Perimeter, s.Eccentricity, s.Solidity, s.Extent, s.MajorAxisLength, s.MinorAxisLength];
    end

    features = [features; featureVector];

    % Labeling
    if contains(lower(imageFiles(i).name), 'tumor')
        labels = [labels; 1]; % Tumor present
    else
        labels = [labels; 0]; % No tumor
    end
end

% normalization 
features = normalize(features);

%% set training and test
cv = cvpartition(size(features,1),'HoldOut',0.2);
idx = cv.test;

XTrain = features(~idx,:);
YTrain = labels(~idx);
XTest = features(idx,:);
YTest = labels(idx);

%% neural network matlab function
net = patternnet(10); % 10 hidden neurons

% Setup training parameters
net.trainFcn = 'trainscg'; % Scaled conjugate gradient
net.performFcn = 'crossentropy'; % For classification
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train
[net,tr] = train(net, XTrain', YTrain');

%
%test
YPred = net(XTest');
YPred = round(YPred); % Threshold output at 0.5

accuracy = sum(YPred' == YTest)/numel(YTest)*100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

% Confusion Matrix
figure;
plotconfusion(categorical(YTest), categorical(YPred'))
title('Confusion Matrix for Brain Tumor Detection');

