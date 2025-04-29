% Tumor detection B-spline Curve Fitting

% Shape-focused, refine boundary detection 


% 1.- Load/Read the image 
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp','All Image Files'});
filepath = fullfile(pathname, filename);
img = imread(filepath);

% 2.- Grayscale 
imgray = rgb2gray(img);

% 3.- Denoise 
imfilt = medfilt2(imgray,[3,3]);

% 4.- Edge detection 
imedges = edge(imfilt,'Canny');

% 5.- Morphology ops
imedges = imclose(imedges,strel('disk',3));
imedges = imfill(imedges,'holes');
imedges = bwareaopen(imedges,500);

% 6.- Boundaries 
boundaries = bwboundaries(imedges);

% 7.- Choose max region (can change based on threshold)
numPixels = cellfun(@numel, boundaries);
[~, idx] = max(numPixels);
boundary = boundaries{idx};

x = boundary(:,2);
y = boundary(:,1);

figue('color','w')
imshow(imgray)
hold on
plot(x,y,'g','LineWidth',1.5);

%%%%%
% 8.- B-Spline Curve Fitting 

t = linspace(0,1,length(x));
spline_order = 4; %(degree + 1)
knotsnumber = round(length(x)/5);
knots = linspace(0,1,knotsnumber);
knots = augknt(knots, spline_order);

spx = spap2(knots, spline_order, t, x');
spy = spap2(knots, spline_order, t, y');
tpoints = linspace(0,1,500);
xx = fnval(spx, tpoints);
yy = fnval(spy, tpoints);

figure('Color','w')
imshow(img)
hold on
plot(xx,yy,'r-','LineWidth',2);
