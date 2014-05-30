function [ hist_color ] = color_histogram( img, img_path, bin_size )
%COLOR_HISTOGRAM Summary of this function goes here
%   Detailed explanation goes here

    try
        % try to load features
        load( img_path, 'hist_color');
    catch

        img_r = img(:,:,1);
        img_g = img(:,:,2);
        img_b = img(:,:,3);
        
        hist_r = imhist( img_r(:), bin_size );
        hist_g = imhist( img_g(:), bin_size );
        hist_b = imhist( img_b(:), bin_size );
        
        hist_color = [hist_r'/sum(hist_r), hist_g'/sum(hist_g), hist_b'/sum(hist_b)];
        
        save( img_path , 'hist_color');
    end
end

