%clear variables
clc; clear; close all;

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

VOCopts.testset='test';
% VOCopts.testset='val';

results_auc = pascal_classifier( VOCopts );

latex_table( VOCopts.classes, results_auc, 1 );