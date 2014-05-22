function generate_dictionary_global( )
%GENERATE_DICTIONARY Extract SIFT descriptors from all images to forma
%visual dictionary
%   Loop through all available images to extract SIFT descriptors and form
%   a visual dictionary

    % change this path if you install the VOC code elsewhere
    addpath([cd '/VOCcode']);

    % initialize VOC options
    VOCinit;

    cls_clusters = [VOCopts.dictpath_global, 'dictionary.mat'];
    if exist(cls_clusters, 'file')
        disp('     [Dictionary exist !]')
        return
    end
        
    % extract features
    detector = extract_features( VOCopts );	

    % clusterize!!
    make_cluster( VOCopts, 500 );
    disp('     [Done !]')


%%   
% Extract features from all images
%_
function detector = extract_features(VOCopts)


    file_list = dir( sprintf( VOCopts.imgpath, '*' ) );
    tic;
    
    % iterate through all feature files
    for file = 1 : length( file_list )
        
        file_path = fullfile( VOCopts.imgpathfolder, file_list( file ).name );

        % display progress
        if toc>1
            fprintf('%s: extracting features: %d/%d\n',file, length(file_list));
            drawnow;
            tic;
        end
        

        % extract features for image
        try
            % try to load features
            load( sprintf( [VOCopts.dictpath_global VOCopts.dictnamefrmt] ,file_list( file ).name) ,'fd');
        catch
            % compute and save features
            I = imread( file_path );
            fd = extractfd( VOCopts,  I);

            save( sprintf( [VOCopts.dictpath_global VOCopts.dictnamefrmt] ,file_list( file ).name) ,'fd');

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
    cls_folder = sprintf( VOCopts.dictclasspath_global, cls );
    cls_clusters = [VOCopts.dictpath_global, cls, '.mat'];

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
