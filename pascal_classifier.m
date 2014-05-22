function pascal_classifier
clc; clear; close all;
% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% train and test classifier for each class
for i=1:VOCopts.nclasses
    
    cls=VOCopts.classes{i};
    
    
    % train classifier
    classifier = train(VOCopts, cls);                  
    
    % test classifier
    test(VOCopts,cls,classifier);                   
    
    % compute and display ROC
    [fp,tp,auc]=VOCroc(VOCopts,'comp1',cls,true);   
    
    if i<VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        pause;
    end
    
end

function [ dictionary ] = load_dictionary( VOCopts, cls )
%LOAD_DICTIONARY Summary of this function goes here
%   Detailed explanation goes here

    disp(['==== Dictionary for class ' cls])

    cls_clusters = [VOCopts.dictpath, cls, '.mat'];
    
    dictionary = [];
    if exist(cls_clusters, 'file')
        load( [VOCopts.dictpath, cls, '.mat'] );
        dictionary = centroids;
    else
        disp('     [Dictionary NOT FOUND !]')   
    end
        


% train classifier
function classifier = train( VOCopts, cls, dictionary)

    % load 'train' image set for class
    [ids,classifier.gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');

    % load dictionary for the class
    classifier.dictionary = load_dictionary( VOCopts, cls );
    
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
            fd = extractfd( VOCopts, I, classifier.dictionary);
            
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        
        classifier.FD( 1:length(fd) , i ) = fd';
        
    end

function histogram = make_histogram( image_feat, dict_feat )
   
    histogram = zeros( 1, size( dict_feat, 2 ) );
    
    % find the best sift descriptor matching
    matches = siftmatch ( image_feat, dict_feat );
    
    for i = 1 : size( dict_feat,2 )
        idx = matches(2,:) == i;
        histogram(i) = sum( matches(1,idx) );
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

    
% return L2 norm as the similarity measurement
function d = similarity_measurement(f1, f2)
    d = norm( f1 - f2 );

% run classifier on test images
function test(VOCopts, cls, classifier)

    % load test set ('val' for development kit)
    [ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

    % create results file
    fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

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
            fd=extractfd(VOCopts, I, classifier.dictionary);
            
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end

        % compute confidence of positive classification
        c=classify(VOCopts,classifier,fd');

        % write to results file
        fprintf(fid,'%s %f\n',ids{i},c);
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
    fd = make_histogram_2( fd, dictionary' );
    %[xx,yy] = stairs(histogram);
    %area(xx,yy);


%sift feature extractor
function fd = sift_descriptor(I)

    I = double(rgb2gray(I)/256) ;
    [frames,descriptors] = sift(I, 'Verbosity', 1) ;

    perm = randperm( size(frames,2) ) ;
    sel  = perm(1:50);

    fd = descriptors(:,sel);
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
