function fd = texture_cooccurrence(img, img_path)
%TEXTURE_FEATURES Extract texture features from an image
%   Using the MATLAB functions graycomatrix and graycoprops, texture
%   features of a matrix are extracted, with orientation 0 degrees and
%   distance two. The statistics extracted are Contrast, Homogeneity,
%   Energy and Correlation


    try
        % try to load features
        load( img_path, 'fd');
    catch

        % compute and save features
        img = rgb2gray(img);
        % orientation: 0 degrees, distance: 2
        comat_im = graycomatrix(img, 'Offset',[0 2], 'Symmetric', true);
        % statistics
        fd = graycoprops(comat_im,{'Contrast','Homogeneity', 'Energy', 'Correlation'});
        fd = struct2array( fd );
    
        save( img_path , 'fd');
        
        % clf; imagesc(I) ; colormap gray ; axis image ; hold on ;
        % h=plotsiftframe( frames(:,sel),'style','arrow' ) ; set(h,'LineWidth',1,'Color','g') ;
        % h=plot(frames(1,sel),frames(2,sel),'r.');
    end

    
end