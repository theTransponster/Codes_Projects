function bb = extractBoundaries(pointSet,ft)

xbp = pointSet(:,1);
ybp = pointSet(:,2);

% 1.- Extract max and min vertex 
minx = min(xbp); maxx = max(xbp);
miny = min(ybp); maxy = max(ybp);

% 2.- Make mesh of points 
interval_x = minx:ft:maxx;
interval_y = miny:ft:maxy;
[bigGridX, bigGridY] = meshgrid(interval_x, interval_y);

% 3.- Extract mesh points that are inside the polygon boundary
[in1,on1] = inpolygon(bigGridX(:),bigGridY(:),xbp,ybp);
pointsInt = [bigGridX(in1),bigGridY(in1)];
%
gX = bigGridX(:);
gY = bigGridY(:);

[idxm, locb] = ismember(pointsInt,[gX,gY],'rows');
allidx = 1:length(gX);
notidxm = setdiff(allidx,locb);
gX(notidxm) = 1;
gY(notidxm) = 1;
gX(locb) = 0;
gY(locb) = 0;
gX2 = reshape(gX,size(bigGridX));
gY2 = reshape(gY,size(bigGridY));
boundaries = bwboundaries(~gX2);
% scale each boundary
bb = {};
for i = 1:1:length(boundaries)
    indx = boundaries{i,1};
    vecp = [];
    for j = 1:1:length(indx)
        vecp = [vecp; bigGridX(indx(j,1),indx(j,2)), bigGridY(indx(j,1),indx(j,2))];
    end
    
    bb{i,1} = vecp;
end







end