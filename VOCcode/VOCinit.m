clear VOCopts
warning('off','all')


% get current directory with forward slashes
addpath('../lib/sift')
addpath('./functions/')
addpath('./features/')

% add weka path
if strncmp(computer,'PC',2)
    javaaddpath('C:\Program Files\Weka-3-7\weka.jar');
    run('../lib\vlfeat-0.9.18\toolbox\vl_setup')
elseif strncmp(computer,'GLNX',4)
    javaaddpath('/home/evargasv/weka-3-7-11/weka.jar');
    run('../lib/vlfeat-0.9.18/toolbox/vl_setup')
elseif strncmp(computer,'MACI64',3)
    javaaddpath('/Applications/weka-3-7-11-apple-jvm.app/Contents/Resources/Java/weka.jar')
    run('../lib/vlfeat-0.9.18/toolbox/vl_setup')
end

import weka.*;


cwd=cd;
cwd(cwd=='\')='/';

% change this path to point to your copy of the PASCAL VOC data
VOCopts.datadir=[cwd '/'];

% change this path to a writable directory for your results
VOCopts.resdir=[cwd '/results/'];
if ~exist( VOCopts.resdir, 'dir' )
    mkdir( VOCopts.resdir );
end

% change this path to a writable local directory for the example code
VOCopts.localdir=[cwd '/local/'];
if ~exist( VOCopts.localdir, 'dir' )
    mkdir( VOCopts.localdir );
end

% initialize the test set

VOCopts.testset='val'; % use validation data for development test set
% VOCopts.testset='test'; % use test set for final challenge

% initialize paths

VOCopts.imgsetpath=[VOCopts.datadir 'VOC2006/ImageSets/%s.txt'];
VOCopts.clsimgsetpath=[VOCopts.datadir 'VOC2006/ImageSets/%s_%s.txt'];
VOCopts.annopath=[VOCopts.datadir 'VOC2006/Annotations/%s.txt'];
VOCopts.imgpathfolder=[VOCopts.datadir 'VOC2006/PNGImages/'];
VOCopts.imgpath=[VOCopts.imgpathfolder '%s.png'];
VOCopts.clsrespath=[VOCopts.resdir '%s_cls_' VOCopts.testset '_%s.txt'];
VOCopts.detrespath=[VOCopts.resdir '%s_det_' VOCopts.testset '_%s.txt'];


% initialize the VOC challenge options

VOCopts.classes={'bicycle','bus','car','cat','cow','dog',...
                'horse','motorbike','person','sheep'};
% VOCopts.classes={'bus'};

VOCopts.nclasses=length(VOCopts.classes);
VOCopts.minoverlap=0.5;


% initialize example options

VOCopts.dictpath = [VOCopts.localdir 'dictionary/'];
VOCopts.dictclasspath = [VOCopts.dictpath ,'%s/'];

VOCopts.dictpath_global = [VOCopts.localdir 'dictionary_global/'];
if ~exist( VOCopts.dictpath_global, 'dir' )
    mkdir( VOCopts.dictpath_global );
end

VOCopts.dictclasspath_global = [VOCopts.dictpath_global ,'%s/'];
VOCopts.dictnamefrmt  = '%s_fd.mat';

VOCopts.exfdpath=[VOCopts.localdir '%s_fd.mat'];

% initialize options for our implementation 

VOCopts.fd_folders = {'sift/', 'bg_sift/', 'textures/', 'color/', 'dsift/', 'hsv/'};

for i = 1: length(VOCopts.fd_folders)
    folder = [VOCopts.localdir  VOCopts.fd_folders{i}];
    if ~exist( folder, 'dir' )
        mkdir( folder );
    end
end


VOCopts.sift_path = [VOCopts.localdir VOCopts.fd_folders{1} 'sift_%d_%s.mat'];
VOCopts.hist_path = [VOCopts.localdir VOCopts.fd_folders{2} 'hist%d_%d_%s.mat'];
VOCopts.text_path = [VOCopts.localdir VOCopts.fd_folders{3} 'text_%d_%s.mat'];
VOCopts.color_path = [VOCopts.localdir VOCopts.fd_folders{4} 'color%d_%d_%s.mat'];
VOCopts.dsift_path = [VOCopts.localdir VOCopts.fd_folders{5} 'dsift_%d_%s.mat'];
VOCopts.dhist_path = [VOCopts.localdir VOCopts.fd_folders{5} 'dhist%d_%d_%s.mat'];
VOCopts.hsv_path   = [VOCopts.localdir VOCopts.fd_folders{6} 'hsv%d_%d_%s.mat'];
