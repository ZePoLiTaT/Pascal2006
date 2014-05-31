function [ fd ] = dsift_features( img, img_path, bin_size, magnif )
%SIFT_DENSE Summary of this function goes here
%   Detailed explanation goes here

    try
        % try to load features
        load( img_path, 'fd');
    catch

        if(~exist('bin_size','var'))
            bin_size = 8;
        end
        if(~exist('magnif','var'))
            magnif = 3;
        end
        
        img = single(vl_imdown(rgb2gray(img))) ;
        img_smt = vl_imsmooth(img, sqrt((bin_size/magnif)^2 - .25)) ;

        fd = vl_dsift(img_smt, 'step', 2) ;
        fd(3,:) = bin_size/magnif ;
        fd(4,:) = 0 ;
        
        save( img_path , 'fd');
        
        % clf; imagesc(I) ; colormap gray ; axis image ; hold on ;
        % h=plotsiftframe( frames(:,sel),'style','arrow' ) ; set(h,'LineWidth',1,'Color','g') ;
        % h=plot(frames(1,sel),frames(2,sel),'r.');
    end
    
end

