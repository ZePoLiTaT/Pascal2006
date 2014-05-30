function feature = gray_mean( im )
%GRAY_MEAN Computes the gray channel of an image
%   Transform an image in gray scale, if the image is in RGB and compute
%   the mean of the intensities.

    if size(im,3) > 1,
    	im = rgb2gray(im);
    end

    feature = mean(im);
end