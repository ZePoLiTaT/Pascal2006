function v = computeFeatureVector(A, fd_num)
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



















