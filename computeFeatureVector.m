function v = computeFeatureVector(A)
%
% Describe an image A using texture features.
%   A is the image
%   v is a 1xN vector, being N the number of features used to describe the
% image
%


%% gray mean:
%v = gray_mean( A );

%% texture features (CHECorr):
%v = texture_features(A);
%v = normalize_features(v);

%% texture features (HC):
%v = texture_features(A);
%v = [v(2)];
%v = normalize_features(v);

%% rgb mean:
%v = rgb_mean( A );

%% rgb mean + texture features:
% v = [texture_features(A) rgb_mean( A )];
% v = [rgb_mean( A ) texture_features(A)];
% v = normalize_features(v);


%% CIELab mean

%v = CIELab_mean( A );

%% CIELab + texture features

%v = [CIELab_mean( A ) texture_features(A)];
%v = normalize_features(v);

%% SIFT Features
v = sift_descriptor(A);
v = normalize_features(v);

end

function feature = gray_mean( im )
%GRAY_MEAN Computes the gray channel of an image
%   Transform an image in gray scale, if the image is in RGB and compute
%   the mean of the intensities.

    if size(im,3) > 1,
    	im = rgb2gray(im);
    end

    feature = mean(im);
end

function stats_im = texture_features(im)
%TEXTURE_FEATURES Extract texture features from an image
%   Using the MATLAB functions graycomatrix and graycoprops, texture
%   features of a matrix are extracted, with orientation 0 degrees and
%   distance two. The statistics extracted are Contrast, Homogeneity,
%   Energy and Correlation

    im = rgb2gray(im);
    % orientation: 0 degrees, distance: 2
    comat_im = graycomatrix(im, 'Offset',[0 2], 'Symmetric', true);
    % statistics
    stats_im = graycoprops(comat_im,{'Contrast','Homogeneity', 'Energy', 'Correlation'});
    stats_im = struct2array( stats_im );
 end

function feature = rgb_mean( im)
%RGB_MEAN Computes the mean of each channel from an RGB image
%   The image is reshaped in order to compute the mean value of each
%   channel and a vector of dimensions 1 x 3 is returned

    [R,C,~] = size(im);
    im = reshape(im, R*C, 3);
    feature = mean(im);
end

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

function feature = mean_rgb_patch(im)
%MEAN_RGB_PATCH

    feature = [];
    [nr,nc,nz] = size(im);
    for i=1:10
    	for j=1:10
    		dv = im( floor(1+(i-1)*nr/10) : floor(i*nr/10),...
                     floor(1+(j-1)*nc/10) : floor(j*nc/10), : );
    		feature = [feature;sum(sum(double(dv)))/(size(dv,1)*size(dv,2))];
        end
    end
end


function feature = sift_descriptor( im )
%SIFT_DESCRIPTOR Extracts SIFT features over a given image

    I = double( rgb2gray(im)/256 ) ;
    [~,feature] = sift(I, 'Verbosity', 1) ;
    % clf; imagesc(I) ; colormap gray ; axis image ; hold on ;
    % h=plotsiftframe( frames(:,sel),'style','arrow' ) ; set(h,'LineWidth',1,'Color','g') ;
    % h=plot(frames(1,sel),frames(2,sel),'r.');
end






