img = rgb2gray(imread('MRI1.jpg'));
imshow(img)
template = rgb2gray(imread('MRI_TUMOR.jpg')); % use already tumor shape recognized previoulsy as template input
imshow(template)
correlationMap = normxcorr2(template,img);
[maxCorr, maxIndex] = max(correlationMap(:));
[yPeak, xPeak] = ind2sub(size(correlationMap),maxIndex);

yOffset = yPeak - size(template,1);
xOffset = xPeak - size(template,2);
figure
imshow(img)
rectangle('Position',[xOffset, yOffset, size(template,2), size(template,1)],...
    'EdgeColor','r','LineWidth',2);