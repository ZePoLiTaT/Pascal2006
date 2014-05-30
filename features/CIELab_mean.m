function feature = CIELab_mean( im )
%CIELab_mean Converts an RGB image into CIELab color space and computes the
%mean value of each channel
%   The image is reshaped in order to compute the mean value of each
%   channel and a vector of dimensions 1 x 3 is returned

    colorTransform = makecform('srgb2lab');
    im = applycform(im, colorTransform);
    [R,C,~] = size(im);
    im = reshape(im, R*C, 3);
    feature = mean(im);

end