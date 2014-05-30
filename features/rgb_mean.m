function feature = rgb_mean( im)
%RGB_MEAN Computes the mean of each channel from an RGB image
%   The image is reshaped in order to compute the mean value of each
%   channel and a vector of dimensions 1 x 3 is returned

    [R,C,~] = size(im);
    im = reshape(im, R*C, 3);
    feature = mean(im);
end