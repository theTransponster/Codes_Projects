% Diverticulitis detection CT analysis PRE-PROCESSING 
% Pre-version1, version that can be shared**


% load image (slice by slice) 
imD = imread('Image-0.jpg');
imshow(imD)

% grayscale 
grayImg = rgb2gray(imD);

% remove noise
filteredImg = medfilt2(grayImg, [3 3]);

% binarization and segmentation 
bw = imbinarize(filteredImg, 'adaptive', 'Sensitivity', 0.5);
bw_clean = imopen(bw, strel('disk', 2));
bw_clean = imclose(bw_clean, strel('disk', 4));

% classify objects by geometrical props 
props = regionprops(bw_clean, 'Area', 'Eccentricity', 'BoundingBox');

% Identify objects  
% change threshold before training network 
imshow(grayImg); hold on;
for i = 1:length(props)
    if props(i).Area < 40 && props(i).Area > 20 && props(i).Eccentricity < 0.95
        rectangle('Position', props(i).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
    end
end



