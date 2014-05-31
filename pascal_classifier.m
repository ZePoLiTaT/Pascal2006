function results_auc = pascal_classifier( VOCopts )
%PASCAL_CLASSIFIER

    
    % create an AUC results table 
    results_auc = zeros( VOCopts.nclasses, 1 );
    

    % train and test classifier for each class
    for i=1:VOCopts.nclasses

        cls=VOCopts.classes{i};
        
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                           LOAD DICTIONARY
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % If the selected parameter was to use a sparse dictionary for the
        % class then load regular sift dictionary
        if ( VOCopts.params.sparse_size(i) ~= -1 )
            sift_dict = load_dictionary( VOCopts, VOCopts.params.sparse_size(i) , '' );
            
        % If dense region selection was chosen for the class, load the
        % dense sift dictionary
        elseif ( VOCopts.params.dense_size(i) ~= -1 )
            sift_dict = load_dictionary( VOCopts, VOCopts.params.dense_size(i), 'dsift_' );
            
        % another feature different to BoW was selected
        else
            sift_dict = [];
        end

        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                       FEATURE EXTRACTION
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % load features
        train_dataset = load_features( VOCopts, i, 'train', sift_dict, VOCopts.params.use_bbox(i) );
        test_dataset = load_features( VOCopts, i, VOCopts.testset, sift_dict, false );
        
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                            TRAIN
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % train classifier
        classifier = train_bounding_box(VOCopts, i, cls, train_dataset);                  

        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                             TEST
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % test classifier
        test(VOCopts, cls, classifier, test_dataset);

        % compute and display ROC
        figure(i);
        [fp, tp, results_auc(i) ] = VOCroc(VOCopts,'comp1',cls,true);  
    end
    
    
end

%--------------------------------------------------------------------------
function dictionary = load_dictionary( VOCopts, dict_size, suffix )
%LOAD_DICTIONARY Summary of this function goes here
%   Detailed explanation goes here

    % folder where the features will be stored
    centroids_file = [VOCopts.dictpath_global, ['centroids_' suffix num2str(dict_size) '.mat']];

    try
        load(centroids_file);
        dictionary = centroids;
    catch
        disp('Dictionary could NOT be loaded!! ')
    end
end

%--------------------------------------------------------------------------
function dataset = load_features( VOCopts, cls_ix, stage, sift_dict, use_bbox )

    % initialize dataset
    dataset.gt_ix = [];
    dataset.FD = [];
    
    % get the size of the dictionaries
    dict_size = size(sift_dict, 2);

    % set the number of offsets and directions of the co-occurrence
    % matrices
    offset = create_offsets( [2,5,30], [1 1 1 1] );
    img_divs = 10;
    
    % load images for the given stage: train, validate or classify
    features_file = sprintf(VOCopts.clsimgsetpath, VOCopts.classes{1}, stage);
    [ids, ~] = textread(features_file,'%s %d');

    % Bounding box of the feature
    bounding_box = {};
    
    tic;
    for i=1:length(ids)

        % display progress
        if toc > 1
            fprintf('Loading [%s]: %d/%d\n',stage,i,length(ids));
            drawnow;
            tic;
        end
        
        % read annotation for the given train image
        rec = PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                           LOAD REGION 
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
        % use bounding boxes if we can use annotations
        if use_bbox
            % find objects of class and extract difficult flags for these objects
            diff = [rec.objects.difficult];
            % extract bounding boxes for non-difficult objects    
            bounding_box{end+1} = cat( 1 , rec.objects(~diff).bbox)';
        % otherwise, use the whole image size as the analysis box
        else
            bounding_box{end+1} = cat( 1 , [1 1 rec.imgsize(1) rec.imgsize(2)]' );
        end
        
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                         FEATURE EXTRACTION
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Extract features on the patches of the bounding boxes
        for j=1: size( bounding_box{end}, 2)
                
            % image boxes
            cur_box = bounding_box{end}(:,j);
            img  = imread(sprintf(VOCopts.imgpath,ids{i}));
            img_box  = img( cur_box(2):cur_box(4), cur_box(1):cur_box(3), : );
            
            % Name of the feature image
            if use_bbox; iix=j; else iix=99; end

            % Initialize feature vectors
            fd_hist = [];
            fd_color = [];
            fd_text = [];
            
            % -------------------- BAG OF FEATURES -----------------------
            
            % If SPARSE features were selected for the class
            if ( VOCopts.params.sparse_size( cls_ix ) ~= -1 )
                sift_path = sprintf(VOCopts.sift_path, iix, ids{i});
                hist_path = sprintf(VOCopts.hist_path, dict_size, iix, ids{i});
                
                fd_sift = sift_features( img_box, sift_path );
                fd_hist = sift_histogram( fd_sift, sift_dict, hist_path);
                
            % If DENSE features were selected for the class
            elseif ( VOCopts.params.dense_size( cls_ix ) ~= -1 )
                sift_path = sprintf(VOCopts.dsift_path, iix, ids{i});
                hist_path = sprintf(VOCopts.dhist_path, dict_size, iix, ids{i});
                
                fd_sift = sift_features( img_box, sift_path );
                fd_hist = sift_histogram( fd_sift, sift_dict, hist_path);
            end
            
            % ------------------------- COLOR ---------------------------
            
            % If HSV color was selected
            if ( VOCopts.params.hsv( cls_ix ) ~= -1 )
                hsv_path = sprintf(VOCopts.hsv_path, img_divs, iix, ids{i});
                
                img_hsv = rgb2hsv(img_box);
                fd_color = mean_rgb_patch( img_hsv, img_divs, hsv_path );
                fd_color = normalize_features( fd_color );
                
            elseif ( VOCopts.params.rgb( cls_ix ) ~= -1 )
                rgb_path = sprintf(VOCopts.color_path, img_divs, iix, ids{i});
                fd_color = mean_rgb_patch( img_box, img_divs, rgb_path );
                fd_color = normalize_features( fd_color );
            end
            
            
            % ------------------------- TEXTURE  ---------------------------
            if ( VOCopts.params.coocmat( cls_ix ) ~= -1 )
                text_path = sprintf(VOCopts.text_path, iix, ids{i});
                fd_text = texture_cooccurrence( img_box, text_path, offset );
                fd_text = normalize_features( fd_text );
            end
            
            
            
            % Concatenate ALL selected features
            fd = [fd_hist, fd_color, fd_text];
            
            dataset.gt_ix = [ dataset.gt_ix; i ];
            dataset.FD = [ dataset.FD, fd'];
        
        end
    end
end

%--------------------------------------------------------------------------
function offset = create_offsets( dir, ang_idx )
    
    % vector with all available offsets
    offset_all = [0 1; -1 1; -1 0; -1 -1];
    
    % which angles do we actually want
    ang_idx = find(ang_idx == true);
    
    % work only with the angles specified by ang
    offset_all = offset_all( ang_idx, : );
    ang_num = size(offset_all,1);
    
    % initialize output with size:  ( #dir x #angles , 2 )
    offset = [];
    
    for i = 1: length(dir)
        offset = [offset; offset_all * dir(i)];
    end
end

%--------------------------------------------------------------------------
% train classifier
function classifier = train_bounding_box( VOCopts, i, cls, dataset)

    % load 'train' image set for class
    [ids, gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
    
    classifier.FD = dataset.FD;
    classifier.gt = gt( dataset.gt_ix );
    
    classifier = weka_classifier( classifier.FD', classifier.gt, VOCopts.params.classifier(i) );
end  
 

%--------------------------------------------------------------------------
% run classifier on test images
function test(VOCopts, cls, classifier, dataset)

    % load test set ('val' for development kit)
    [ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');


    evaluator.FD = dataset.FD;
    evaluator.gt = gt( dataset.gt_ix );
    
    [correct,prob] = weka_evaluation( classifier, evaluator.FD', evaluator.gt );

    % save results 
    % create results file
    fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

    for i=1:length(ids)
        fprintf(fid,'%s %f\n',ids{i},prob(i));
    end

    % close results file
    fclose(fid);
end

