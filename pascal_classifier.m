function example_classifier
clc; clear; close all;
% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% add weka path
if strncmp(computer,'PC',2)
    javaaddpath('C:\Program Files\Weka-3-7\weka.jar');
elseif strncmp(computer,'GLNX',4)
    javaaddpath('/home/evargasv/weka-3-7-11/weka.jar');
elseif strncmp(computer,'MACI64',3)
    javaaddpath('/Applications/weka-3-7-11-apple-jvm.app/Contents/Resources/Java/weka.jar')
end


import weka.*;


% initialize VOC options
VOCinit;


% load dictionary for the class 
dictionary = load_dictionary( VOCopts, 500 );

% train and test classifier for each class
for i=1:VOCopts.nclasses
    
    cls=VOCopts.classes{i};
    
    
    % train classifier
    classifier = train_bounding_box(VOCopts, cls, dictionary);                  
    
    % test classifier
    test(VOCopts,cls,dictionary,classifier);                   
    
    % compute and display ROC
    [fp,tp,auc]=VOCroc(VOCopts,'comp1',cls,true);   
    
    if i<VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        pause;
    end
    
end

function [ dictionary ] = load_dictionary( VOCopts, num_clusters )
%LOAD_DICTIONARY Summary of this function goes here
%   Detailed explanation goes here


    % folder where the features will be stored
    centroids_file = [VOCopts.dictpath_global, ['centroids_' num2str(num_clusters) '.mat']]

    try
        load(centroids_file);
        dictionary = centroids;
    catch
        disp('Dictionary could NOT be loaded!! ')
    end


% train classifier
function classifier = train_bounding_box( VOCopts, cls, dictionary)

    % load 'train' image set for class
    [ids,classifier.gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');

    % Bounding box of the feature
    bounding_box = {};
    
    % extract features for each image
    classifier.FD=zeros(0,length(ids));
    tic;
    for i=1:length(ids)

        % display progress
        if toc > 1
            fprintf('%s: train: %d/%d\n',cls,i,length(ids));
            drawnow;
            tic;
        end
        
        % read annotation for the given train image
        rec = PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
        
        % find objects of class and extract difficult flags for these objects
        
        diff = [rec.objects.difficult];

        % extract bounding boxes for non-difficult objects    
        bounding_box{end+1} = cat( 1 , rec.objects(~diff).bbox)';
        
        try
            % try to load features
            load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            
            % compute and save features
            I  = imread(sprintf(VOCopts.imgpath,ids{i}));
            fd = [];
            
            %Extract features on the patches of the bounding boxes
            for j=1: size( bounding_box{end}, 2)
                cur_box = bounding_box{end}(:,j);
                img_box = I( cur_box(2):cur_box(4), cur_box(1):cur_box(3), : );

                fd_box = sift_descriptor( img_box );
                
                fd = [ fd,  fd_box ];
            end
            
            fd = make_histogram_3( fd, dictionary );
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        
        classifier.FD( 1:length(fd) , i ) = fd';
        
    end
classifier = weka_classifier( classifier.FD', classifier.gt );
    
    
    
% train classifier
function classifier = train( VOCopts, cls, dictionary)

    % load 'train' image set for class
    [ids,classifier.gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');

    
    % extract features for each image
    classifier.FD=zeros(0,length(ids));
    tic;
    for i=1:length(ids)

        % display progress
        if toc > 1
            fprintf('%s: train: %d/%d\n',cls,i,length(ids));
            drawnow;
            tic;
        end

        try
            
            % try to load features
            load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            
            % compute and save features
            I  = imread(sprintf(VOCopts.imgpath,ids{i}));
            fd = extractfd( VOCopts, I, dictionary);
            
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        
        classifier.FD( 1:length(fd) , i ) = fd';
        
    end
classifier = weka_classifier( classifier.FD', classifier.gt );
    

function histogram = make_histogram( image_feat, dict_feat )
   
    histogram = zeros( 1, size( dict_feat, 2 ) );
    
    % find the best sift descriptor matching
    matches = siftmatch ( image_feat, dict_feat );
    
    for i = 1 : size( matches,2 )
        idx = matches(2,i);
        histogram(idx) = histogram(idx) + 1;
    end
    
function histogram = make_histogram_2( image_feat, dict_feat )
    
    threshold = 0.7;
    histogram = zeros( 1, size( dict_feat, 2 ) );
    
    for i = 1 : size( dict_feat, 2 )
        
        %initialize distance with high values
        d = zeros(1, size( image_feat, 2 ));
        for j = 1 : size( image_feat, 2 ) 
            %euclidan distance between features
            d(j) = similarity_measurement( dict_feat(:,i), image_feat(:,j) );
        end
            
        %closest match
        [min_d, min_ix] = min(d);
        
        %increment histogram only when the value is greater than a given
        %threshold
        if ( min_d <= threshold )
            histogram(min_ix) = histogram(min_ix) + 1;
        end
    end

function histogram = make_histogram_3( features, dictionary )
   
    histogram = zeros( 1, size( dictionary, 2 ) );
    
    for i = 1:size(features,2)
        fd = features(:,i);
        d = sum(fd.*fd)+sum(dictionary.*dictionary)-2*fd'*dictionary;
        [d_min, ix_min] = min(d);
        histogram(ix_min) = histogram(ix_min) + 1;
    end
        
    
% return L2 norm as the similarity measurement
function d = similarity_measurement(f1, f2)
    d = norm( f1 - f2 );

% run classifier on test images
function test(VOCopts,cls,dictionary,classifier)

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

% extract features for each image
vecTest=zeros(0,length(ids));

% classify each image
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd = extractfd( VOCopts, I, dictionary);
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        
    end

    vecTest(1:length(fd),i)=fd;
    
    % compute confidence of positive classification
    %c=classify(VOCopts,classifier,fd);
    
    % write to results file
    %fprintf(fid,'%s %f\n',ids{i},c);
end

[correct,prob] = weka_evaluation( classifier, vecTest', gt );
correct

% save results 

% create results file
fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

for i=1:length(ids)
    fprintf(fid,'%s %f\n',ids{i},prob(i));
end

% close results file
fclose(fid);

% trivial feature extractor: compute mean RGB
function fd = extractfd(VOCopts, I, dictionary)

    % fd = [];
    % [nr,nc,nz] = size(I);
    % for i=1:10,
    % 	for j=1:10,
    % 		dv = I(floor(1+(i-1)*nr/10) : floor(i*nr/10),...
    %                floor(1+(j-1)*nc/10) : floor(j*nc/10),:);
    % 		fd = [fd;sum(sum(double(dv)))/(size(dv,1)*size(dv,2))];
    % %fd=squeeze(sum(sum(double(I)))/(size(I,1)*size(I,2)));
    % 	end
    % end
    % 
    % fd = fd(:);
    % fd = [fd; sift_descriptor(I)];
    
    fd = sift_descriptor(I);
    fd = make_histogram_3( fd, dictionary );
    %[xx,yy] = stairs(histogram);
    %area(xx,yy);


%sift feature extractor
function descriptors = sift_descriptor(I)

    I = double(rgb2gray(I)/256) ;
    [~,descriptors] = sift(I, 'Verbosity', 1) ;

    %fd = descriptors;
    %fd = fd(:);

    % clf; imagesc(I) ; colormap gray ; axis image ; hold on ;
    % h=plotsiftframe( frames(:,sel),'style','arrow' ) ; set(h,'LineWidth',1,'Color','g') ;
    % h=plot(frames(1,sel),frames(2,sel),'r.');


% trivial classifier: compute ratio of L2 distance betweeen
% nearest positive (class) feature vector and nearest negative (non-class)
% feature vector
function c = classify(VOCopts,classifier,fd)

    d=sum(fd.*fd)+sum(classifier.FD.*classifier.FD)-2*fd'*classifier.FD;
    dp=min(d(classifier.gt>0));
    dn=min(d(classifier.gt<0));
    c=dn/(dp+eps);
