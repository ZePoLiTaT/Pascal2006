function generate_dictionary_dsift
%GENERATE_DICTIONARY Extract SIFT descriptors from all images to forma
%visual dictionary
%   Loop through all available images to extract SIFT descriptors and form
%   a visual dictionary

    clc; clear; close all;

    % change this path if you install the VOC code elsewhere
    addpath([cd '/VOCcode']);

    % initialize VOC options
    VOCinit;
    
    % extract the ids of the images to use for the dictionary generation
    ids = get_subset_ids(VOCopts);
        
    % extract features
    features = extract_features( VOCopts, ids );
    
    % create dictionaries of sizes 100 to 1000
    %clusters = 100:100:1000;   %clusters for sift
    clusters = [100,300,600,900]; %clusters for dsift
    
    % cluster all the sift feature vectors
    for i = 1: length(clusters)
        
        make_cluster( VOCopts, clusters(i), features );
        disp('     [Done !]')
        
    end
end


function ids = get_subset_ids(VOCopts)
    % get the name of the 1st class
    cls = VOCopts.classes{1};
    
    % load 'train' image set for class
    [ids_train,~] = textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
    
    ids = [ids_train];
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                Extract features from all images
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
function features = extract_features(VOCopts, file_ids)

    features = [];
    tic;
    
    % iterate through all feature files
    for i = 1 : length( file_ids )
        
        file_path = fullfile( sprintf(VOCopts.imgpath, file_ids{i}) );

        % display progress
        if toc>1
            fprintf('Extracting features: %d/%d\n',int16(i), length(file_ids));
            drawnow;
            tic;
        end
        
        img_box = imread( file_path );
        dsift_path = sprintf(VOCopts.dsift_path, 1, file_ids{i} );
        fd_dsift = dsift_features( img_box, dsift_path );
        
        features = [features, fd_dsift];
    end
    
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                               CLUSTERIZE 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
function centroids = make_cluster( VOCopts, num_clusters, features )

    % initialize centroids
    centroids = [];

    % folder where the features will be stored
    centroids_file = [VOCopts.dictpath_global, ['centroids_dsift_' num2str(num_clusters) '.mat']]
  
    try
        load(centroids_file)
    catch
    
        if isempty(features)
            dsift_folder = [VOCopts.localdir VOCopts.fd_folders{5}];

            % Create a temporal folder for the features of the class 
            % before kmeans
            if ~exist( dsift_folder , 'dir')
                disp('DSIFT Features must be created before calling kmeans')
                return
            end

            %if the centroids file is already calculated ..
            file_list = dir( [dsift_folder 'dsift_*.mat'] );

            tic;
            % iterate through all feature files
            for file = 1 : length( file_list )

                % display progress
                if toc>1
                    fprintf('Clustering features: %d/%d [Size: %d] \n',int16(file), length(file_list), size(features,2) );
                    drawnow;
                    tic;
                end

                sift_path = [dsift_folder  file_list(file).name];
                load( sift_path )

                features = [features, fd];

            end
        end
        
        tic
        centroids = vl_kmeans(features,num_clusters); 
        toc
        
        save( centroids_file, 'centroids' );
    end
end