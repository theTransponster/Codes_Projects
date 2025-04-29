%% Catmull-Rom Spline algorithm 

% 1.- Define data points to fit and establish total number
PointsSet = [1,2; 3, 4; 5, 7; 8, 5; 9, 3; 11, 6; 12, 8];
Nps = length(PointsSet);
figure(1)
plot(PointsSet(:,1),PointsSet(:,2),'or',MarkerSize=10)

% 2.- Define number of points to include per segment and generate points
% vector parameter t
NpsCR = 5;
t = linspace(0,1,NpsCR);


% 3.- Make iterations to fit segments between 4 points (3 different)
pointTt = [];
CRp = [];
for i = 1:1:Nps-1
    % 3.1.- Establish control points 
    p0 = PointsSet(max(i-1,1),:);
    p1 = PointsSet(i,:);
    p2 = PointsSet(i+1,:);
    p3 = PointsSet(min(i+2,Nps),:);
    for j = 1:1:length(t)
        pointT = (1/2)*((2*p1)+((-p0+p2)*t(j))+(((2*p0)-(5*p1)+(4*p2)-p3)*(t(j)^2))+((-p0+(3*p1)-(3*p2)+p3)*(t(j)^3)));
        pointTt = [pointTt; pointT];
    end
    CRp = [CRp; pointTt];
    pointTt = [];
end

hold on
plot(CRp(:,1),CRp(:,2),'.-b',MarkerSize=10)