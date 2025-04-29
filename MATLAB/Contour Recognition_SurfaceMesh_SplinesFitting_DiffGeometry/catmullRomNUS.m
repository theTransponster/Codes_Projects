function CRsp = catmullRomNUS(PointsSet,varargin)


Nps = length(PointsSet);

if isempty(varargin{1})
    NpsCR = 5;
else
    NpsCR = varargin{1};
end

pointTt = [];
CRp = [];
for i = 1:1:Nps-1

    % Establish control points 
    p0 = PointsSet(max(i-1,1),:);
    p1 = PointsSet(i,:);
    p2 = PointsSet(i+1,:);
    p3 = PointsSet(min(i+2,Nps),:);
    %t = linspace(0,1,NpsCR);
    pair = [p1;p2];
    dist = pdist(pair,'euclidean');
    nm = floor(dist/NpsCR);
    t = linspace(0,1,nm);
    for j = 1:1:length(t)
        pointT = (1/2)*((2*p1)+((-p0+p2)*t(j))+(((2*p0)-(5*p1)+(4*p2)-p3)*(t(j)^2))+((-p0+(3*p1)-(3*p2)+p3)*(t(j)^3)));
        pointTt = [pointTt; pointT];
    end
    CRp = [CRp; pointTt(1:end-1,:)];
    pointTt = [];
    %NpsCR = NpsCR + 1;
end
CRsp = CRp;


end