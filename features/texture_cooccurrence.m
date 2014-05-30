function coocurrence = texture_cooccurrence(img, img_path, offset)
%TEXTURE_FEATURES Extract texture features from an image
%   Using the MATLAB functions graycomatrix and graycoprops, texture
%   features of a matrix are extracted, with orientation 0 degrees and
%   distance two. The statistics extracted are Contrast, Homogeneity,
%   Energy and Correlation


    try
        % try to load features
        load( img_path, 'coocurrence');
    catch

        % compute and save features
        img = rgb2gray(img);
        % orientation: 0 degrees, distance: 2
        comat_im = graycomatrix(img, 'Offset',offset, 'Symmetric', true);
        % statistics
        coocurrence = graycoprops(comat_im,{'Contrast','Homogeneity', 'Energy', 'Correlation'});
        coocurrence = struct2array( coocurrence );
    
        save( img_path , 'coocurrence');
        
        % clf; imagesc(I) ; colormap gray ; axis image ; hold on ;
        % h=plotsiftframe( frames(:,sel),'style','arrow' ) ; set(h,'LineWidth',1,'Color','g') ;
        % h=plot(frames(1,sel),frames(2,sel),'r.');
    end

    
end