function [ fd ] = sift_features( img, img_path )
%SIFT_FEATURES Extracts SIFT features over a given image
%   Detailed explanation goes here

    try
        % try to load features
        load( img_path, 'fd');
    catch

        % compute and save features
        img = double( rgb2gray(img) )/256 ;
        [~,fd] = sift(img, 'Verbosity', 1);

        save( img_path , 'fd');
        
        % clf; imagesc(I) ; colormap gray ; axis image ; hold on ;
        % h=plotsiftframe( frames(:,sel),'style','arrow' ) ; set(h,'LineWidth',1,'Color','g') ;
        % h=plot(frames(1,sel),frames(2,sel),'r.');
    end
end

