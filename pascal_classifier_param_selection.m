function results_auc = pascal_classifier( VOCopts, dict_size, is_sparse )
%PASCAL_CLASSIFIER

    
    % create an AUC results table 
    results_auc = zeros( VOCopts.nclasses, 1 );
    
    
    % load dictionary depending if it is sparse or dense
    if is_sparse
        dic_suffix = '';
    else
        dic_suffix = 'dsift_';
    end
    sift_dict = load_dictionary( VOCopts, dict_size, dic_suffix );
   
    % load features
    train_dataset = load_features( VOCopts, 'train', sift_dict, is_sparse, is_sparse );
    test_dataset = load_features( VOCopts, VOCopts.testset, sift_dict, is_sparse, false );

    % train and test classifier for each class
    %for i=1:VOCopts.nclasses
    for i=1:VOCopts.nclasses

        cls=VOCopts.classes{i};


        % train classifier
        classifier = train_bounding_box(VOCopts, cls, train_dataset);                  

        % test classifier
        test(VOCopts, cls, classifier, test_dataset);

        % compute and display ROC
        figure(i);
        [fp, tp, results_auc(i) ] = VOCroc(VOCopts,'comp1',cls,true);   

%         if i<VOCopts.nclasses
%             fprintf('press any key to continue with next class...\n');
%             pause;
%         end

    end
    
    
end


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

function dataset = load_features( VOCopts, stage, sift_dict, sparse_feat, use_bbox )

    % initialize dataset
    dataset.gt_ix = [];
    dataset.FD = [];
    
    % get the size of the dictionaries
    dict_size = size(sift_dict, 2);

    % set the number of offsets and directions of the co-occurrence
    % matrices
    offset = create_offsets( [2,5,30], [1 1 1 1] );
    img_divs = 10;
    
    % load 'train' image set for class
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
        
        
        %Extract features on the patches of the bounding boxes
        for j=1: size( bounding_box{end}, 2)
                
            cur_box = bounding_box{end}(:,j);
            
            img  = imread(sprintf(VOCopts.imgpath,ids{i}));
            img_box  = img( cur_box(2):cur_box(4), cur_box(1):cur_box(3), : );
            
            if use_bbox; iix=j; else iix=99; end

            % Initialize feature vectors
            fd_hist = [];
            fd_text = [];
            fd_color = [];
            
            
            % Paths to feature directories
            if sparse_feat
                sift_path = sprintf(VOCopts.sift_path, iix, ids{i});
                hist_path = sprintf(VOCopts.hist_path, dict_size, iix, ids{i});
            else
                sift_path = sprintf(VOCopts.dsift_path, iix, ids{i});
                hist_path = sprintf(VOCopts.dhist_path, dict_size, iix, ids{i});
            end
            
            
            text_path = sprintf(VOCopts.text_path, iix, ids{i});
            rgb_path = sprintf(VOCopts.color_path, img_divs, iix, ids{i});
            hsv_path = sprintf(VOCopts.hsv_path, img_divs, iix, ids{i});
            
            % SIFT features
            fd_sift = sift_features( img_box, sift_path );
            fd_hist = sift_histogram( fd_sift, sift_dict, hist_path);
            
            % Texture features
%             fd_text = texture_cooccurrence( img_box, text_path, offset );
%             fd_text = normalize_features( fd_text );
            
            % Color features
            % RGB
%             fd_color = mean_rgb_patch( img_box, img_divs, rgb_path );
%             fd_color = normalize_features( fd_color );
            % HSV
%             img_hsv = rgb2hsv(img_box);
%             fd_color = mean_rgb_patch( img_hsv, img_divs, hsv_path );
%             fd_color = normalize_features( fd_color );
            
            % Concatenate features using DENSE SIFT
            fd = [fd_hist, fd_text, fd_color];
            
            dataset.gt_ix = [ dataset.gt_ix; i ];
            dataset.FD = [ dataset.FD, fd'];
            
            %classifier.FD( 1:length(fd) , i ) = fd';
        
        end
        
    end
    
end

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

% train classifier
function classifier = train_bounding_box( VOCopts, cls, dataset)

    % load 'train' image set for class
    [ids, gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
    
    classifier.FD = dataset.FD;
    classifier.gt = gt( dataset.gt_ix );
    
    classifier = weka_classifier( classifier.FD', classifier.gt );
end  
 

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

