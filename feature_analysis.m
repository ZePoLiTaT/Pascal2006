%clear variables
clc; clear; close all;

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% Sizes for SPARSE SIFT
%  dict_sizes = [100:100:1000];
%  dict_sizes = 100;

% Sizes for DENSE SIFT
 dict_sizes = [100,300,600,900];

% create an AUC results table 
results_auc = zeros( VOCopts.nclasses, length(dict_sizes) );

for i = 1: length(dict_sizes)
    results_auc(:, i) = pascal_classifier( VOCopts, dict_sizes(i), false );
end

latex_table( VOCopts.classes, results_auc, dict_sizes );

%TODO:
% 1. Spatial pyramid
% 2. Separate features and classifier per class
% 3. Dense sift