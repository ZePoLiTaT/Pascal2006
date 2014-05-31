clc; clear; close all;

car = '000017';
bycicle = '000013';
cow = '000019';

img_src = imread(sprintf('VOC2006/PNGImages/%s.png',cow) );


% img = img_box;
figure(1); clf; imagesc(img); colormap gray ; axis image ; hold on ;

% ----------------- SPARSE SIFT - ANDREA
img = double( rgb2gray(img) )/256 ;
[f, fd_sift] = sift(img, 'Verbosity', 0); 

perm = randperm(size(f,2)) ;
sel = 1:size(f,2); %perm(1:400);
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

h=plotsiftframe( f(:,sel),'style','arrow' ) ; set(h,'LineWidth',1,'Color','g') ;
h=plot(f(1,sel),f(2,sel),'r.');


% ----------------- SPARSE SIFT - VLFEAT
img = img_src;
%test....
% img = img_box;

figure(1); clf; imagesc(img); colormap gray ; axis image ; hold on ;
img = single(rgb2gray(img)) ;
[f, d] = vl_sift(img) ;
perm = randperm(size(f,2)) ;
sel = 1:size(f,2); %perm(1:100) ; %
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;
%h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
%set(h3,'color','g') ;


% ----------------- DENSE SIFT - VLFEAT
img = img_src;
img = single(vl_imdown(rgb2gray(img_src))) ;
binSize = 8 ;
magnif = 3 ;
Is = vl_imsmooth(img, sqrt((binSize/magnif)^2 - .25)) ;


[f, d] = vl_dsift(Is, 'step', 10) ;

f(3,:) = binSize/magnif ;
f(4,:) = 0 ;
 

perm = randperm(size(f,2)) ;
sel = 1:size(f,2); %perm(1:50) ;
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

%h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
%set(h3,'color','g') ;