
% Count blue cells (DAPI) and green cells (GFAP protein) 

% Read Image
img = imread('GFAP1.jpg');  

% Separate color channels
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

% Convert to double for processing
R = double(R);
G = double(G);
B = double(B);

%%%%%% BLUE CELLS
blueMask = (B > 100) & (R < 100) & (G < 100); % Thresholds CAN CHANGE 

% Clean up mask
blueMask = bwareaopen(blueMask, 5); % Remove small noise

% Label and count blue cells
blueLabels = bwlabel(blueMask);
numBlueCells = max(blueLabels(:));

%%%%%%% GREEN CELLS 
greenMask = (G > 100) & (R < 100) & (B < 100); % Thresholds may need adjustment

% Clean up mask
greenMask = bwareaopen(greenMask, 5);

% Label and count green cells
greenLabels = bwlabel(greenMask);
numGreenCells = max(greenLabels(:));

% DISPLAY COUNTS 
fprintf('Number of blue cells: %d\n', numBlueCells);
fprintf('Number of green cells: %d\n', numGreenCells);

% Optional: visualize results
figure;
subplot(1,3,1); imshow(img); title('Original Image');
subplot(1,3,2); imshow(blueMask); title('Blue Cells');
subplot(1,3,3); imshow(greenMask); title('Green Cells');
