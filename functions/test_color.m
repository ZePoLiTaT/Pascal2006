clc; clear; close all;

car = '000017';
bycicle = '000013';
cow = '000019';

img_src = imread(sprintf('VOC2006/PNGImages/%s.png',cow) );
figure(99); imshow(img_src);

%RGB
img = img_src;
figure(1); subplot(1,3,1), imagesc(img(:,:,1)), subplot(1,3,2), imagesc(img(:,:,2)), subplot(1,3,3), imagesc(img(:,:,3))

%HSV
img = rgb2hsv(img_src);
figure(2); subplot(1,3,1), imagesc(img(:,:,1)), subplot(1,3,2), imagesc(img(:,:,2)), subplot(1,3,3), imagesc(img(:,:,3))

%CIELAB
colorTransform = makecform('srgb2lab');
img = applycform(img_src, colorTransform);

figure(3); subplot(1,3,1), imagesc(img(:,:,1)), subplot(1,3,2), imagesc(img(:,:,2)), subplot(1,3,3), imagesc(img(:,:,3))