function norm_features = normalize_features( features )
%% NORM_FEATURES Normalizes the feature matrix 
%   NORM_FEATURES(features) returns a normalized version of the features
%   where the mean value of each feature is 0 and the standard deviation is
%   one.
    
    mean_f = mean(features);
    norm_features = bsxfun( @minus, features, mean_f );
    
    std_f = std(features);
    norm_features = bsxfun( @rdivide, norm_features, std_f );
    
end