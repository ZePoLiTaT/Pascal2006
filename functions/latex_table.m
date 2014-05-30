function   latex_table( classes, auc )
%LATEX_TABLE Summary of this function goes here
%   Detailed explanation goes here

    dicts_size = size( auc, 2 );

    res_frmt = ['%f '];
    res_frmt = repmat( res_frmt,  1, dicts_size );
    res_frmt = ['%s & ' res_frmt];
    
    
    disp('\begin{table}[H]')
    disp('\centering')
    disp('\begin{tabular}{|c|c|}')
    disp('\hline')
    disp('\rowcolor{black} \color{white} \textbf{Class} & \color{white} \textbf{AUC} \\')
    disp('\hline')
    
    for i=1:length(classes)
        
        disp( sprintf(res_frmt, classes{i}, auc(i,:)) )
        
    end
    
    disp('\hline')
    disp('\end{tabular}')
    disp(sprintf('\\caption{Dictionary Size: %d words}'))
    disp(sprintf('\\label{tb:eval}'))
    disp('\end{table}')
end

