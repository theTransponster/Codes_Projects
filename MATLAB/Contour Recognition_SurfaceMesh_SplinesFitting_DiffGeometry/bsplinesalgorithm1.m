%% B-splines algorithm 
% For reference of graphic visualization visit https://people.eecs.berkeley.edu/~sequin/CS284/IMGS/bsplinebasics.gif

% 1.- Define control points 
P = [1,3; 4,6; 8, 4; 9, 2; 10,5; 11,6;12,3]; 
plot(P(:,1),P(:,2),'o-',MarkerSize=10)

% 2.- Define number of points, degree of b-spline and calculate number of
% knots
nP = length(P);
degBs = 3;
numknots = nP + degBs + 1;

% 3.- Create Knot vector 
optkv = 2;
% Option 1; uniform clamped knot vector between 0 and 1
% Option 2; uniform clamped knot vector between specified points 
if optkv == 1
    knotVector = [zeros(1,degBs+1), linspace(0,1,numknots-2*(degBs + 1)), ones(1,degBs+1)];
elseif optkv == 2
    % specified knotvector endpoints
    n1 = 2;
    n2 = 7;
    knotVector = [repmat(n1,degBs+1,1).',linspace(n1,n2,numknots-2*(degBs + 1)), repmat(n2,degBs+1,1).'];
end
