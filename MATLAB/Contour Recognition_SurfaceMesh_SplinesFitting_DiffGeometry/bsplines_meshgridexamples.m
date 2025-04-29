n = 3;
divs = 430;
t1 = -2:(3--2)/divs:3;
%t = [ -2 -1 0 0.25 0.5 0.75 1 2 3 ];
xpb = [xt{1,1}(:,1); xt{2,1}(:,1)];
ypb = [xt{1,1}(:,2); xt{2,1}(:,2)];
zpb = [xt{1,1}(:,3); xt{2,1}(:,3)];
P = [xpb.';ypb.';zpb.'];
%P = [ 0.1993 0.4965 0.6671 0.7085 0.6809 0.2938 ...
%    ; 0.8377 0.8436 0.7617 0.6126 0.212 0.1067 ...
%    ; 0.987 0.2378 0.4512 0.3421 0.212 0.184];
X = bspline_deboor(n,t1,P);
figure;
hold all;
plot3(X(1,:), X(2,:), X(3,:), '.r');
plot3(P(1,:), P(2,:), P(3,:), '-kv');
hold off;

%%
F = TR.ConnectivityList;
V = TR.Points;
xp = pointsu(:,1);
yp = pointsu(:,2);
zp = pointsu(:,3);
xmin = min(xp)-1; xmax = max(xp)+1;
ymin = min(yp)-1; ymax = max(yp)+1;
nx = 100;  ny = 100; % define resolution of grids
xi = linspace(xmin,xmax,nx);
yi = linspace(ymin,ymax,ny);
%hold on
[X,Y] = meshgrid(xi,yi);

Z = gridtrimesh(F,V,X,Y);
surf(X,Y,Z)
hold on
%plot3(xp,yp,zp,'*')

%%
%x = XI(in); 
%y = YI(in);
x = xp;
y = yp;
z = zp;
bb = {};
auxc = 1;
while auxc<3
    k = boundary(x,y,0.9);
    x1 = x(k); y1 = y(k); z1 = z(k);
    x(k) = []; y(k) = [];
    bb{auxc,1} = [x1,y1];
    hold on
    plot3(x1,y1,z1,'.-')
    auxc = auxc + 1;
    hold on
    plot3(xp,yp,zp,'.')
end

%%
xp1 = xt{1,1}(:,1); yp1 = xt{1,1}(:,2); zp1 = xt{1,1}(:,3);
[in,on] = inpolygon(XI(:),YI(:),xp1,yp1);
puntos1 = [XI(in), YI(in); XI(on), YI(on)];
%plot(xp1,yp1,'.-r')
hold on
plot(XI(in),YI(in),'+g')
hold on
plot(XI(on),YI(on),'.b')
kk = boundary(puntos1(:,1),puntos1(:,2));
hold on
plot(puntos1(kk,1),puntos1(kk,2),'.-b')


