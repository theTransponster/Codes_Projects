% Algorithm to calculate a Bezier curve 

% 1.- Define your control points (nx2 for 2D, or nx3 for 3D)
P = [1,2; 3,6; 5,4; 8, 10];
% plot the control points 
plot(P(:,1),P(:,2),'*')

% 2.- Define number of points in the curve (parameter t)
np = 100;

% 3.- Make vector of parameter t
t = linspace(0,1,np);

% 4.- Initialize vector to store the new curve points 
Bzcurve = zeros(np,2);

% 5.- Generate new points for the curve
for i = 1:np
    Bzcurve(i, :) = (1 - t(i))^3 * P(1,:) + ...
              3 * (1 - t(i))^2 * t(i) * P(2,:) + ...
              3 * (1 - t(i)) * t(i)^2 * P(3,:) + ...
              t(i)^3 * P(4,:);
end

% 6.- Plot new curve
hold on
plot(Bzcurve(:,1),Bzcurve(:,2),'.-')

%% Case for more than 4 control points (higher degree, non cubic bezier curve)

% 1.- Establish number of control points
ncp = 6;

% 2.- Create vector of control points 
vcp = [1,2; 4,6; 7, 2; 4,9; 6,7; 8,3 ];
plot(vcp(:,1),vcp(:,2),'-.',MarkerSize = 20)

% 3.- Define number of points in Bezier curve (for t) 
npB = 20;

% 4.- Make vector for t
t = linspace(0,1,npB);

% 5.- Initialize vector to store the new bezier curve 
Bzcurve = zeros(npB,2);

% 6.- Obtain binomial coefficients 
bcoff = zeros(ncp,1);
auxcoff_i = 0;
auxcoff_n = ncp-1;
for i = 1:1:ncp
    bcoff(i) = factorial(auxcoff_n)/(factorial(auxcoff_i)*factorial(auxcoff_n-auxcoff_i));
    auxcoff_i = auxcoff_i + 1;
end

% 7.- Calculate the points for the Bezier curve
Bzaux = [0,0];
for i = 1:1:npB
    for j = 1:1:ncp
        Bzaux1 = bcoff(j)*((1-t(i))^(ncp-1-(j-1)))*(t(i)^(j-1))*vcp(j,:);
        Bzaux = Bzaux + Bzaux1;
    end
    Bzcurve(i,:) = Bzaux;
    Bzaux = [0,0];
end
hold on
plot(Bzcurve(:,1),Bzcurve(:,2),'*-',MarkerSize=13)

%% Case for arbitrary number of control points and for 3D points 
% 1.- Establish number of control points
ncp = 6;

% 2.- Create vector of control points 
vcp = [1,2,1; 4,6,3; 7, 2,4; 4,9,3; 6,7,4; 8,3,2 ];
plot3(vcp(:,1),vcp(:,2),vcp(:,3),'-.',MarkerSize = 20)
%
% 3.- Define number of points in Bezier curve (for t) 
npB = 20;

% 4.- Make vector for t
t = linspace(0,1,npB);

% 5.- Initialize vector to store the new bezier curve 
Bzcurve = zeros(npB,3);

% 6.- Obtain binomial coefficients 
bcoff = zeros(ncp,1);
auxcoff_i = 0;
auxcoff_n = ncp-1;
for i = 1:1:ncp
    bcoff(i) = factorial(auxcoff_n)/(factorial(auxcoff_i)*factorial(auxcoff_n-auxcoff_i));
    auxcoff_i = auxcoff_i + 1;
end

% 7.- Calculate the points for the Bezier curve
Bzaux = [0,0,0];
for i = 1:1:npB
    for j = 1:1:ncp
        Bzaux1 = bcoff(j)*((1-t(i))^(ncp-1-(j-1)))*(t(i)^(j-1))*vcp(j,:);
        Bzaux = Bzaux + Bzaux1;
    end
    Bzcurve(i,:) = Bzaux;
    Bzaux = [0,0,0];
end
hold on
plot3(Bzcurve(:,1),Bzcurve(:,2),Bzcurve(:,3),'*-',MarkerSize=13)

