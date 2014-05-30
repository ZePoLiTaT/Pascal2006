function generate_dictionary_global( )
%GENERATE_DICTIONARY Extract SIFT descriptors from all images to forma
%visual dictionary
%   Loop through all available images to extract SIFT descriptors and form
%   a visual dictionary

    % change this path if you install the VOC code elsewhere
    addpath([cd '/VOCcode']);

    % initialize VOC options
    VOCinit;
    
    ids = get_subset_ids(VOCopts);

    cls_clusters = [VOCopts.dictpath_global, 'dictionary.mat'];
    if exist(cls_clusters, 'file')
        disp('     [Dictionary exist !]')
        return
    end
        
    % extract features
    extract_features( VOCopts, ids );	

    % clusterize!!
    %cluserts = [400,600,650];
    cluserts = [100,200,300,700,1000];
    for i = 1: length(cluserts)
        make_cluster( VOCopts, cluserts(i) );
        disp('     [Done !]')
    end
    


function ids = get_subset_ids(VOCopts)
    % get the name of the 1st class
    cls = VOCopts.classes{1};
    % load 'train' image set for class
    [ids_train,~] = textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
    % load 'train' image set for class
    [ids_val,~] = textread(sprintf(VOCopts.clsimgsetpath,cls,'val'),'%s %d');
    % load 'train' image set for class
    [ids_test,~] = textread(sprintf(VOCopts.clsimgsetpath,cls,'test'),'%s %d');
    
    ids = [ids_train; ids_val; ids_test];
    
%%   
% Extract features from all images
%_
function extract_features(VOCopts, file_ids)

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
        

        % extract features for image
        try
            % try to load features
            load( sprintf( [VOCopts.dictpath_global VOCopts.dictnamefrmt] ,file_ids{i}) ,'fd');
        catch
            % compute and save features
            I = imread( file_path );
            fd = extractfd( VOCopts,  I);

            save( sprintf( [VOCopts.dictpath_global VOCopts.dictnamefrmt] ,file_ids{i}) ,'fd');

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


function centroids = make_cluster( VOCopts, num_clusters )

    % initialize centroids
    features = [];
    centroids = [];

    % folder where the features will be stored
    centroids_file = [VOCopts.dictpath_global, ['centroids_' num2str(num_clusters) '.mat']]
  
    try
        load(centroids_file)
    catch
    
        % Create a temporal folder for the features of the class 
        % before kmeans
        if ~exist( VOCopts.dictpath_global , 'dir')
            disp('Features must be created before calling kmeans')
            return
        end


        %if the centroids file is already calculated ..
        file_list = dir( [VOCopts.dictpath_global '*_fd.mat'] );

        tic;
        
        % iterate through all feature files
        for file = 1 : length( file_list )

            % display progress
            if toc>1
                fprintf('Clustering features: %d/%d [Size: %d] \n',int16(file), length(file_list), size(features,2) );
                drawnow;
                tic;
            end
            
            file_path = fullfile( VOCopts.dictpath_global, file_list( file ).name );
            load( file_path )

            features = [features, fd];

        end
        
        centroids = vl_kmeans(features,num_clusters); %,'method', 'elkan') ;
        
        
%         opts = statset('UseParallel',1)
%         [~, centroids] = kmeans(features',num_clusters, ...
%                                         'distance','sqEuclidean', ...
%                                         'EmptyAction','singleton', ...
%                                         'Options',opts);
        save( centroids_file, 'centroids' );
    end
