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