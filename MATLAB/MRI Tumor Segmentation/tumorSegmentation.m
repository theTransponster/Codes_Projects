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


