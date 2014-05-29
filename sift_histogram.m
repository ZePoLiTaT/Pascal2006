function [ histogram ] = sift_histogram( features, dictionary, varargin )
%SIFT_HISTOGRAM create a histogram of words based on a SIFT vocabulary
%   Detailed explanation goes here

    histogram = zeros( 1, size( dictionary, 2 ) );
    
    for i = 1:size(features,2)
        fd = features(:,i);
        d = sum(fd.*fd)+sum(dictionary.*dictionary)-2*fd'*dictionary;
        [d_min, ix_min] = min(d);
        histogram(ix_min) = histogram(ix_min) + 1;
    end

    % If additional flag is provided, then plot the histogram
    if numel(varargin) == 1
        figure(99);
        [x,y] = stairs( histogram );
        area(x,y);
    end
    
end

