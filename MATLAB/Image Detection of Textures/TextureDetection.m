clear all
close all

% Based on contrast, homogeneity, correlation, energy, entropy, mean, variance 


%%Alfombra / Rug 
% alfombra1=imread('alfombra1.jpg');
% alfombra2=imread('alfombra3.jpg');
% alf1_gris=rgb2gray(alfombra1);
% alf2_gris=rgb2gray(alfombra2);
% var_alf1=var(var(im2double(alf1_gris)));
% var_alf2=var(var(im2double(alf2_gris)));
% min_alf1=min(min(im2double(alf1_gris)));
% min_alf2=min(min(im2double(alf2_gris)));
% figure(1)
% subplot(2,3,1),imshow(alfombra1), axis off;
% subplot(2,3,2),imshow(alf1_gris),axis off;
% subplot(2,3,3),imhist(alf1_gris),axis off;
% subplot(2,3,4),imshow(alfombra2),axis off;
% subplot(2,3,5),imshow(alf2_gris),axis off;
% subplot(2,3,6),imhist(alf2_gris),axis off; 
% 
% E1=entropy(alf1_gris);
% E2=entropy(alf2_gris);
% s1=graycoprops(alf1_gris);
% s2=graycoprops(alf2_gris);
% LBP4= efficientLBP(alf1_gris);
% LBP5=efficientLBP(alf2_gris);
% figure(2)
% subplot(2,1,1),imshow(LBP4), axis off;
% subplot(2,1,2),imshow(LBP5), axis off;
% 
% %%
% %Ladrillos / Brick
% ladrillos1=imread('ladrillos1.jpg');
% ladrillos2=imread('ladrillos2.jpg');
% lad1_gris=rgb2gray(ladrillos1);
% lad2_gris=rgb2gray(ladrillos2);
% var_lad1=var(var(im2double(lad1_gris)));
% var_lad2=var(var(im2double(lad2_gris)));
% min_lad1=min(min(im2double(lad1_gris)));
% min_lad2=min(min(im2double(lad2_gris)));
% 
% figure(3)
% subplot(2,3,1),imshow(ladrillos1), axis off;
% subplot(2,3,2),imshow(lad1_gris),axis off;
% subplot(2,3,3),imhist(lad1_gris),axis off;
% subplot(2,3,4),imshow(ladrillos2),axis off;
% subplot(2,3,5),imshow(lad2_gris),axis off;
% subplot(2,3,6),imhist(lad2_gris),axis off; 
% 
% E3=entropy(lad1_gris);
% E4=entropy(lad2_gris);
% s3=graycoprops(lad1_gris);
% s4=graycoprops(lad2_gris);
% LBP2= efficientLBP(lad1_gris);
% LBP3=efficientLBP(lad2_gris);
% figure(4)
% subplot(2,1,1),imshow(LBP2), axis off;
% subplot(2,1,2),imshow(LBP3), axis off;
% 
% %%[g1cm,SI]=graycomatrix(I,'NumLevels',256]
% %%glcm2=graycomatrix(I)
% 
% %%
% %Madera / wood
% madera1=imread('madera1.jpg');
% madera2=imread('madera2.jpg');
% mad1_gris=rgb2gray(madera1);
% mad2_gris=rgb2gray(madera2);
% var_mad1=var(var(im2double(mad1_gris)));
% var_mad2=var(var(im2double(mad2_gris)));
% min_mad1=min(min(im2double(mad1_gris)));
% min_mad2=min(min(im2double(mad2_gris)));
% figure(5)
% subplot(2,3,1),imshow(madera1), axis off;
% subplot(2,3,2),imshow(mad1_gris),axis off;
% subplot(2,3,3),imhist(mad1_gris),axis off;
% subplot(2,3,4),imshow(madera2),axis off;
% subplot(2,3,5),imshow(mad2_gris),axis off;
% subplot(2,3,6),imhist(mad2_gris),axis off; 
% 
% E5=entropy(mad1_gris);
% E6=entropy(mad2_gris);
% s5=graycoprops(mad1_gris);
% s6=graycoprops(mad2_gris);
% LBP= efficientLBP(mad1_gris);
% LBP1=efficientLBP(mad2_gris);
% 
% figure(6)
% subplot(2,1,1),imshow(LBP),axis off;
% subplot(2,1,2),imshow(LBP1),axis off;
% 
% cortar1=imcrop();
% figure(7)
% imshow(cortar1);
%%Brillo, contraste, homogeneidad, entropía, energía, patrón binario local
%%caracterizar el resultado obtenido del PBL con descriptores simples 
%%caracterizar por descriptores simples, elipse, perimetro, relación
%%perimetro 



%%
%% Parameters to train neural network 
%% contrast, homogeneity, correlation, energy, entropy, mean, variance 


im1=rgb2gray(imread('cobre1.jpg'));
im2=rgb2gray(imread('cobre2.jpg'));
im3=rgb2gray(imread('cobre3.jpg'));
var1=var(var(im2double(im1)));
var2=var(var(im2double(im2)));
var3=var(var(im2double(im3)));
min1=mean(mean((im1)));
min2=mean(mean((im2)));
min3=mean(mean((im3)));
E1=entropy(im1);
E2=entropy(im2);
E3=entropy(im3);
co1=graycomatrix(im1);
s1=graycoprops(im1);
s2=graycoprops(im2);
s3=graycoprops(im3);
LBP1=efficientLBP(im1);

