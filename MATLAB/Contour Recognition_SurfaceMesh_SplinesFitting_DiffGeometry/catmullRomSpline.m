function CRp = catmullRomSpline(pointSet, varargin)

% Catmull-Rom Spline Algorithm 

% Input: 
    % pointSet : coordinates 2D nx2
    % smt : number of points to fit in t 

% Output: 
    % CRsp : coordinates x-y of the fitted spline 


Nps = length(pointSet);

% Define number of points to include per segment and generate points in the
% vector parameter t
if isempty(varargin)
    NpsCR = 5;
else
    NpsCR = varargin{1};
end

t = linspace(0,1,NpsCR);

% Iterations to fit segments between 4 points
pointTt = [];
CRp = [];
for i = 1:1:Nps-1
    p0 = pointSet(max(i-1,1),:);
    p1 = pointSet(i,:);
    p2 = pointSet(i+1,:);
    p3 = pointSet(min(i+2,Nps),:);
    for j = 1:1:length(t)
        pointT = (1/2)*((2*p1)++((-p0+p2)*t(j))+(((2*p0)-(5*p1)+(4*p2)-p3)*(t(j)^2))+((-p0+(3*p1)-(3*p2)+p3)*(t(j)^3)));
        pointTt = [pointTt; pointT];
    end
    CRp = [CRp; pointTt];
    pointTt = [];
end





end