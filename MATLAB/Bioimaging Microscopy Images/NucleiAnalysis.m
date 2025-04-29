% Nuclei analysis 

% Count number of nuclei, extract nuclei polarization (ellipsoid fitting
% major/minor axis) and calculation of nuclei orientation (angle of major
% axis with respect to horizontal axis) 

clear all 

% Read Image
img = imread('DAPI1.png'); 
imgray = rgb2gray(img);


% Binarization 
bw = imbinarize(imgray, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.4);
bw = bwareaopen(bw, 50); % Remove small noise (adjust size threshold if needed)

% Label connected components
labeledImage = bwlabel(bw);

% Measure properties
props = regionprops(labeledImage, 'Centroid', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Eccentricity');

% Nuclei count
numNuclei = numel(props);
fprintf('Number of detected nuclei: %d\n', numNuclei);


% Loop over each nucleus
for k = 1:numNuclei
    % Get properties
    centroid = props(k).Centroid;
    majorAxis = props(k).MajorAxisLength;
    minorAxis = props(k).MinorAxisLength;
    orientation = props(k).Orientation; % Angle in degrees
    
    % Plot the fitted ellipse
    theta = linspace(0, 2*pi, 100);
    a = majorAxis/2; % Semi-major axis
    b = minorAxis/2; % Semi-minor axis
    X = a*cos(theta);
    Y = b*sin(theta);
    
    % Rotation matrix
    R = [cosd(orientation) -sind(orientation); sind(orientation) cosd(orientation)];
    rotatedXY = R * [X; Y];
    
    plot(rotatedXY(1,:) + centroid(1), rotatedXY(2,:) + centroid(2), 'r-', 'LineWidth', 1.5);
    plot(centroid(1), centroid(2), 'b+', 'MarkerSize', 10, 'LineWidth', 1);
    
    % Display orientation angle
    text(centroid(1)+5, centroid(2), sprintf('%.1f^\\circ', orientation), 'Color', 'y', 'FontSize', 8);
    
    % Optional: Print to console
    fprintf('Nucleus %d: Major Axis = %.2f, Minor Axis = %.2f, Orientation = %.2f degrees\n', ...
        k, majorAxis, minorAxis, orientation);
end

hold off;
axis on;
