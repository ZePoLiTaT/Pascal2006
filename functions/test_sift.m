clc; clear; close all;

car = '000017';
bycicle = '000013';
cow = '000019';

img_src = imread(sprintf('VOC2006/PNGImages/%s.png',cow) );
img = single(vl_imdown(rgb2gray(img_src))) ;

figure(1); imagesc(img);

binSize = 8 ;
magnif = 3 ;
Is = vl_imsmooth(img, sqrt((binSize/magnif)^2 - .25)) ;


[f, d] = vl_dsift(Is, 'step', 10) ;

f(3,:) = binSize/magnif ;
f(4,:) = 0 ;
 %[f_, d_] = vl_sift(img, 'frames', f) ;

figure(1); imagesc(Is);
hold on;
perm = randperm(size(f,2)) ;
sel = 1:size(f,2); %perm(1:50) ;
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

%h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
%set(h3,'color','g') ;