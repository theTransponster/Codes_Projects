close all
clear all

% Processing Cell/Histology samples 


celulas=imread('cells(1).jpg');
celulas_gris=rgb2gray(celulas);
umbral=graythresh(celulas_gris);
celulas_byn=im2bw(celulas_gris,umbral);
comp=imcomplement(celulas_byn);
se=strel('disk',1);
dil=imdilate(comp,se);
area=regionprops(dil,'Area');
m=mean([area.Area]);
com=imcomplement(dil);
im=bwareaopen(com, floor(m));

figure(1)
subplot(2,3,1), imshow(celulas_gris), title('Grises'),axis off;
subplot(2,3,2), imshow(celulas_byn), title('Binarizado'), axis off;
subplot(2,3,3), imshow(comp),title('Complemento'), axis off;
subplot(2,3,4), imshow(dil), title('Dilatado'), axis off;
subplot(2,3,5), imshow(com), title('Complemento 2'), axis off;
subplot(2,3,6), imshow(im), title('Bwareaopen'), axis off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
variable=bwmorph(celulas_byn,'Bothat');
variable2=bwmorph(celulas_byn,'Bridge');
variable3=bwmorph(celulas_byn,'Fill');

v1=bwmorph(comp,'Bothat');
v2=bwmorph(comp,'Bridge');
v3=bwmorph(comp,'Fill');

figure(2)
subplot(2,3,1), imshow(variable), title('Binaria BOTHAT'),axis off;
subplot(2,3,2), imshow(variable2),title('Binaria BRIDGE'),axis off;
subplot(2,3,3), imshow(variable3), title('Binaria FILL'),axis off;
subplot(2,3,4), imshow(v1), title('Complemento BOTHAT'),axis off;
subplot(2,3,5), imshow(v2), title('Complemento Bridg'), axis off;
subplot(2,3,6), imshow(v3), title('Complemento FILL'), axis off;

%%%%%%%%%%%%%%%%%%%%%
c2=imread('cells2.jpg');
celulas_gris2=rgb2gray(c2);
umbral2=graythresh(celulas_gris2);
celulas_byn2=im2bw(celulas_gris2,umbral2);
comp2=imcomplement(celulas_byn2);
se=strel('disk',1);
dil2=imdilate(comp2,se);
area2=regionprops(dil2,'Area');
m2=mean([area2.Area]);
com2=imcomplement(dil2);
im2=bwareaopen(com2, floor(m2));

erosion=imerode(comp2,se);
com3=imcomplement(erosion);
area3=regionprops(erosion,'Area');
m3=mean([area3.Area]);
im3=bwareaopen(com3,floor(m3));

figure(3)
subplot(2,3,1), imshow(celulas_gris2), title('Grises'),axis off;
subplot(2,3,2), imshow(celulas_byn2), title('Binarizado'), axis off;
subplot(2,3,3), imshow(comp2),title('Complemento'), axis off;
subplot(2,3,4), imshow(dil2), title('Dilatado'), axis off;
subplot(2,3,5), imshow(com2), title('Complemento 2'), axis off;
subplot(2,3,6), imshow(im2), title('Bwareaopen'), axis off;

figure(4)
subplot(2,3,1), imshow(celulas_gris2), title('Grises'),axis off;
subplot(2,3,2), imshow(celulas_byn2), title('Binarizado'), axis off;
subplot(2,3,3), imshow(comp2),title('Complemento'), axis off;
subplot(2,3,4), imshow(erosion), title('Erosion'), axis off;
subplot(2,3,5), imshow(com3), title('Complemento 2'), axis off;
subplot(2,3,6), imshow(im3), title('Bwareaopen'), axis off;

c2=imread('cells2.jpg');
celulas_gris2=rgb2gray(c2);
umbral2=graythresh(celulas_gris2);
celulas_byn2=im2bw(celulas_gris2,umbral);
comp2=imcomplement(celulas_byn2);
se=strel('disk',1);
dil2=imdilate(comp2,se);
imagen1=imcomplement(dil2);

area2=regionprops(imagen1,'Area');
m2=mean([area2.Area]);
com2=imcomplement(dil2);
im2=bwareaopen(com2, floor(m2));

figure(5)
subplot(2,3,1), imshow(celulas_gris2), title('Grises'),axis off;
subplot(2,3,2), imshow(celulas_byn2), title('Binarizado'), axis off;
subplot(2,3,3), imshow(comp2),title('Complemento'), axis off;
subplot(2,3,4), imshow(dil2), title('Dilatado'), axis off;
subplot(2,3,5), imshow(com2), title('Complemento 2'), axis off;
subplot(2,3,6), imshow(im2), title('Bwareaopen'), axis off;




c2=imread('cells4.jpg');
celulas_gris2=rgb2gray(c2);
umbral2=graythresh(celulas_gris2);
celulas_byn2=im2bw(celulas_gris2,umbral2);
comp2=imcomplement(celulas_byn2);
se=strel('disk',1);
dil2=imdilate(comp2,se);
area2=regionprops(dil2,'Area');
m2=mean([area2.Area]);
com2=imcomplement(dil2);
im2=bwareaopen(com2, floor(m2));

erosion=imerode(comp2,se);
com3=imcomplement(erosion);
area3=regionprops(erosion,'Area');
m3=mean([area3.Area]);
im3=bwareaopen(com3,floor(m3));


figure(6)
subplot(2,3,1), imshow(celulas_gris2), title('Grises'),axis off;
subplot(2,3,2), imshow(celulas_byn2), title('Binarizado'), axis off;
subplot(2,3,3), imshow(comp2),title('Complemento'), axis off;
subplot(2,3,4), imshow(dil2), title('Dilatado'), axis off;
subplot(2,3,5), imshow(com2), title('Complemento 2'), axis off;
subplot(2,3,6), imshow(im2), title('Bwareaopen'), axis off;

figure(7)
subplot(2,3,1), imshow(celulas_gris2), title('Grises'),axis off;
subplot(2,3,2), imshow(celulas_byn2), title('Binarizado'), axis off;
subplot(2,3,3), imshow(comp2),title('Complemento'), axis off;
subplot(2,3,4), imshow(erosion), title('Erosion'), axis off;
subplot(2,3,5), imshow(com3), title('Complemento 2'), axis off;
subplot(2,3,6), imshow(im3), title('Bwareaopen'), axis off;

c2=imread('cells4.jpg');
celulas_gris2=rgb2gray(c2);
umbral2=graythresh(celulas_gris2);
celulas_byn2=im2bw(celulas_gris2,umbral);
comp2=imcomplement(celulas_byn2);
se=strel('disk',1);
dil2=imdilate(comp2,se);
imagen1=imcomplement(dil2);

area2=regionprops(imagen1,'Area');
m2=mean([area2.Area]);
com2=imcomplement(dil2);
im2=bwareaopen(com2, floor(m2));

figure(8)
subplot(2,3,1), imshow(celulas_gris2), title('Grises'),axis off;
subplot(2,3,2), imshow(celulas_byn2), title('Binarizado'), axis off;
subplot(2,3,3), imshow(comp2),title('Complemento'), axis off;
subplot(2,3,4), imshow(dil2), title('Dilatado'), axis off;
subplot(2,3,5), imshow(com2), title('Complemento 2'), axis off;
subplot(2,3,6), imshow(im2), title('Bwareaopen'), axis off;
