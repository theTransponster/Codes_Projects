% Cell Migration Tracking 


clc; clear all;

% Load set of images 
imageFolder = '\migration24hours'; % Set your image folder here
imageFormat = '*.jpg';               % Image format
imageFiles = dir(fullfile(imageFolder, imageFormat));
nImages = length(imageFiles);


maxDistance = 20; % maximum distance (pixels) between frames to be considered the same cell

% Initialize struct to save info
tracks = struct('Centroid', {}, 'TrackID', {}, 'Frames', {});
nextTrackID = 1;

% Process each image
for i = 1:nImages
    % Read image
    img = imread(fullfile(imageFolder, imageFiles(i).name));
    
    % Preprocessing
    if size(img, 3) == 3
        imgGray = rgb2gray(img); % Convert to grayscale if RGB
    else
        imgGray = img;
    end
    imgBW = imbinarize(imgGray, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.4);
    imgBW = bwareaopen(imgBW, 20); % Remove small noise

    % Find centroids
    stats = regionprops(imgBW, 'Centroid');
    centroids = cat(1, stats.Centroid);
    
    % Match centroids to existing tracks
    if i == 1
        % First frame, initialize tracks
        for j = 1:size(centroids,1)
            tracks(j).Centroid = centroids(j,:);
            tracks(j).TrackID = nextTrackID;
            tracks(j).Frames = i;
            nextTrackID = nextTrackID + 1;
        end
    else
        % Subsequent frames
        costMatrix = zeros(length(tracks), size(centroids,1)) + Inf;
        for t = 1:length(tracks)
            for c = 1:size(centroids,1)
                costMatrix(t,c) = norm(tracks(t).Centroid(end,:) - centroids(c,:));
            end
        end
        
        [assignment, unassignedTracks, unassignedCentroids] = assignDetectionsToTracks(costMatrix, maxDistance);
        
        % Update matched tracks
        for a = 1:size(assignment,1)
            trackIdx = assignment(a,1);
            centroidIdx = assignment(a,2);
            tracks(trackIdx).Centroid(end+1,:) = centroids(centroidIdx,:);
            tracks(trackIdx).Frames(end+1) = i;
        end
        
        % Create new tracks for unmatched centroids
        for uc = 1:length(unassignedCentroids)
            idx = unassignedCentroids(uc);
            newTrack.Centroid = centroids(idx,:);
            newTrack.TrackID = nextTrackID;
            newTrack.Frames = i;
            tracks(end+1) = newTrack; 
            nextTrackID = nextTrackID + 1;
        end
    end
    
    % Plot
    %figure(1);
    %imshow(imgGray); hold on;
    %plot(centroids(:,1), centroids(:,2), 'r+');
    %title(['Frame ', num2str(i)]);
    %pause(0.1);
end

% Observe final Trayectories 
figure;
hold on;
colors = lines(nextTrackID-1);
for t = 1:length(tracks)
    plot(tracks(t).Centroid(:,1), tracks(t).Centroid(:,2), '-', 'Color', colors(tracks(t).TrackID,:), 'LineWidth', 2);
end
xlabel('X'); ylabel('Y');
title('Cell Tracks');
grid on;
axis equal;
