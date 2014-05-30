function fd_color_patch = mean_rgb_patch(img, num_divs, img_path)
%MEAN_RGB_PATCH

    try
        % try to load features
        load( img_path, 'fd_color_patch');
    catch
        
        fd_color_patch = [];
        [nr,nc,nz] = size(img);
        for i = 1:num_divs
            for j = 1:num_divs
                dv = img( floor(1+(i-1)*nr/num_divs) : floor(i*nr/num_divs),...
                         floor(1+(j-1)*nc/num_divs) : floor(j*nc/num_divs), : );
                fd_color_patch = [fd_color_patch;sum(sum(double(dv)))/(size(dv,1)*size(dv,2))];
            end
        end
        
        load( img_path, 'fd_color_patch');
    end
end