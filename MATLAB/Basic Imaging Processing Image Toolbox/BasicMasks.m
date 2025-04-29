
% Manual mask 
img = imread('Shapes.jpg');
imshow(img)
poligono = drawpolygon
mask = createMask(poligono);
imshow(mask)

%%

% Mask based on HSV color space
imgHSV = rgb2hsv(img);
gR = [0.2 0.4 0.3 1 0.2 1];
bR = [0.55 0.7 0.4 1 0.4 1];
yR = [ 0.12 0.18 0.5 1 0.5 1];
rR = [0 0.06 0.5 1 0.2 1];
rR2 = [ 0.9 1 0.5 1 0.2 1];

maskG = (imgHSV(:,:,1) >= gR(1) & imgHSV(:,:,1) <= gR(2)) &...
        (imgHSV(:,:,2) >= gR(3) & imgHSV(:,:,2) <= gR(4)) &...
        (imgHSV(:,:,3) >= gR(5) & imgHSV(:,:,3) <= gR(6));
maskB = (imgHSV(:,:,1) >= bR(1) & imgHSV(:,:,1) <= bR(2)) &...
        (imgHSV(:,:,2) >= bR(3) & imgHSV(:,:,2) <= bR(4)) &...
        (imgHSV(:,:,3) >= bR(5) & imgHSV(:,:,3) <= bR(6));
maskY = (imgHSV(:,:,1) >= yR(1) & imgHSV(:,:,1) <= yR(2)) &...
        (imgHSV(:,:,2) >= yR(3) & imgHSV(:,:,2) <= yR(4)) &...
        (imgHSV(:,:,3) >= yR(5) & imgHSV(:,:,3) <= yR(6));
maskR1 = (imgHSV(:,:,1) >= rR(1) & imgHSV(:,:,1) <= rR(2)) &...
        (imgHSV(:,:,2) >= rR(3) & imgHSV(:,:,2) <= rR(4)) &...
        (imgHSV(:,:,3) >= rR(5) & imgHSV(:,:,3) <= rR(6));
maskR2 = (imgHSV(:,:,1) >= rR2(1) & imgHSV(:,:,1) <= rR2(2)) &...
        (imgHSV(:,:,2) >= rR2(3) & imgHSV(:,:,2) <= rR2(4)) &...
        (imgHSV(:,:,3) >= rR2(5) & imgHSV(:,:,3) <= rR2(6));
maskR = maskR1|maskR2;

imshow(maskG+maskB+maskR+maskY)
img2 = imbinarize(maskG+maskB+maskY+maskR);
%%
% ginput color selection
imshow(img)
[x, y] = ginput(1);
row = round(y);
column = round(x);
imgHSV = rgb2hsv(img);
vRef = imgHSV(row, column, 3);
maskV = find(imgHSV(:,:,3) == vRef);