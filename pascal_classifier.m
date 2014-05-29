function pascal_classifier
%PASCAL_CLASSIFIER

    %clear variables
    clc; clear; close all;
    
    % change this path if you install the VOC code elsewhere
    addpath([cd '/VOCcode']);

    
    % initialize VOC options
    VOCinit;

    % create an AUC results table 
    results_auc = zeros( VOCopts.nclasses, 1 );

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
        figure(i);
        [fp, tp, results_auc(i) ]=VOCroc(VOCopts,'comp1',cls,true);   

%         if i<VOCopts.nclasses
%             fprintf('press any key to continue with next class...\n');
%             pause;
%         end

    end
    
    latex_table( VOCopts.classes, results_auc );
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
            load(sprintf(VOCopts.exbgpath,ids{i}),'fd');
        catch
            
            % compute and save features
            I  = imread(sprintf(VOCopts.imgpath,ids{i}));
            fd = [];
            
            %Extract features on the patches of the bounding boxes
            for j=1: size( bounding_box{end}, 2)
                cur_box = bounding_box{end}(:,j);
                img_box = I( cur_box(2):cur_box(4), cur_box(1):cur_box(3), : );

                fd_box = computeFeatureVector( img_box );
                
                fd = [ fd,  fd_box ];
            end
            
            fd = sift_histogram( fd, dictionary, true );
            save(sprintf(VOCopts.exbgpath,ids{i}),'fd');
        end
        
        classifier.FD( 1:length(fd) , i ) = fd';
        
    end
    classifier = weka_classifier( classifier.FD', classifier.gt );
end    
    
    
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
            load(sprintf(VOCopts.exbgpath,ids{i}),'fd');
        catch
            
            % compute and save features
            I  = imread(sprintf(VOCopts.imgpath,ids{i}));
            fd = computeFeatureVector( I );
            fd = sift_histogram( fd, dictionary );
            
            save(sprintf(VOCopts.exbgpath,ids{i}),'fd');
        end
        
        classifier.FD( 1:length(fd) , i ) = fd';
        
    end
    classifier = weka_classifier( classifier.FD', classifier.gt );
end
 

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
            load(sprintf(VOCopts.exbgpath,ids{i}),'fd');
        catch
            % compute and save features
            I=imread(sprintf(VOCopts.imgpath,ids{i}));
            fd = computeFeatureVector(I);
            fd = sift_histogram( fd, dictionary );
            save(sprintf(VOCopts.exbgpath,ids{i}),'fd');

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
end
