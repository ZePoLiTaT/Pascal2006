function [ output_args ] = generate_dictionary( input_args )
%GENERATE_DICTIONARY Extract SIFT descriptors from all images to forma
%visual dictionary
%   Loop through all available images to extract SIFT descriptors and form
%   a visual dictionary

    % change this path if you install the VOC code elsewhere
    addpath([cd '/VOCcode']);

    % initialize VOC options
    VOCinit;


    % train and test detector for each class
    for i=1:VOCopts.nclasses

        % name of the class
        cls=VOCopts.classes{i};

        disp(['==== Dictionary for class ' cls])
        
        cls_clusters = [VOCopts.dictpath, cls, '.mat'];
        if exist(cls_clusters, 'file')
            disp('     [Already exist !]')
            continue
        end
        
        % extract features
        detector = extract_features( VOCopts, cls );	

        % clusterize!!
        make_cluster( VOCopts, cls, 50 );
        disp('     [Done !]')
    end


%%   
% Extract features from all images
%_
function detector = extract_features(VOCopts,cls)

    % folder where the features will be stored
    cls_folder = sprintf( VOCopts.dictclasspath, cls );

    % Create a temporal folder for the features of the class 
    % before kmeans
    if ~exist( cls_folder , 'dir')
         mkdir( cls_folder )
    end

    % load 'train' image set
    [ids,~]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');

    % extract features and bounding boxes
    detector.FD = [];
    detector.bbox = {};
    detector.gt = [];

    for i=1:length(ids)
        % display progress
        if toc>1
            fprintf('%s: extracting features: %d/%d\n',cls,i,length(ids));
            drawnow;
        end

        % read annotation
        rec = PASreadrecord(sprintf(VOCopts.annopath,ids{i}));

        % find objects of class and extract difficult flags for these objects
        clsinds = strmatch(cls,{rec.objects(:).class},'exact');
        diff = [rec.objects(clsinds).difficult];

        % extract bounding boxes for non-difficult objects    
        detector.bbox{end+1}=cat(1,rec.objects(clsinds(~diff)).bbox)';

        % assign ground truth class to image
        if ~isempty(clsinds) && any(~diff)
            gt=1;           % at least one non-difficult object of class
        else
            gt=0;           % only difficult objects
        end

        if gt == 1

            % extract features for image
            try
                % try to load features
                load( sprintf( [cls_folder VOCopts.dictnamefrmt] ,ids{i}) ,'fd');
            catch
                % compute and save features
                I=imread(sprintf(VOCopts.imgpath,ids{i}));
                fd = [];

                for j=1: size( detector.bbox{end}, 2)
                    bbox = detector.bbox{end}(:,j)

                    Iclass = I( bbox(2):bbox(4), ...
                                bbox(1):bbox(3), : );

%                     figure(1); clf; imshow(I); hold on;
%                     rectangle('Position', [bbox(1), bbox(2), bbox(3)-bbox(1), bbox(4)-bbox(2)],...
%                               'EdgeColor' , [1,0,0] );
%                     figure(2); clf; imshow(Iclass);

                    fd = [ fd, extractfd( VOCopts,  Iclass) ];

                end

                save( sprintf( [cls_folder VOCopts.dictnamefrmt] ,ids{i}), 'fd');

            end


        end
    end



% trivial feature extractor: compute mean RGB
function fd = extractfd( VOCopts, I )
    fd = sift_descriptor(I);



%sift feature extractor
function fd = sift_descriptor(I)

    I = double(rgb2gray(I)/256) ;
    [frames,descriptors] = sift(I, 'Verbosity', 1) ;

%     if size(frames,2) > 0
%         clf; imagesc(I) ; colormap gray ; axis image ; hold on ;
%         h=plotsiftframe( frames(:,:),'style','arrow' ) ; set(h,'LineWidth',1,'Color','g') ;
%         h=plot(frames(1,:),frames(2,:),'r.');
%     end

    fd = descriptors;


function centroids = make_cluster( VOCopts, cls, num_clusters )

    % initialize centroids
    features = [];
    centroids = [];

    % folder where the features will be stored
    cls_folder = sprintf( VOCopts.dictclasspath, cls );
    cls_clusters = [VOCopts.dictpath, cls, '.mat'];

    % Create a temporal folder for the features of the class 
    % before kmeans
    if ~exist( cls_folder , 'dir')
         return
    end
    
    %if the centroids file is already calculated ..

    file_list = dir( [cls_folder '*.mat'] );
    
    % iterate through all feature files
    for file = 1 : length( file_list )
        
        file_path = fullfile( cls_folder, file_list( file ).name );
        load( file_path )
        
        features = [features, fd];
        
    end
    
    opts = statset('UseParallel',1)
    [~, centroids] = kmeans(features',num_clusters, ...
                                    'distance','sqEuclidean', ...
                                    'EmptyAction','singleton', ...
                                    'Options',opts);
    save( cls_clusters, 'centroids' );
