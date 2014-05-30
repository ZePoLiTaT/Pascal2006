%clear variables
clc; clear; close all;

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

%dict_sizes = [100:100:700];
dict_sizes = 1000;

% create an AUC results table 
results_auc = zeros( VOCopts.nclasses, length(dict_sizes) );

for i = 1: length(dict_sizes)
    results_auc(:, i) = pascal_classifier( VOCopts, dict_sizes(i) );
end

latex_table( VOCopts.classes, results_auc, dict_sizes );