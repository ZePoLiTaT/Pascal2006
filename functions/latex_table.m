function   latex_table( classes, auc, param_names )
%LATEX_TABLE Summary of this function goes here
%   Detailed explanation goes here

    dicts_size = size( auc, 2 );

    col_frmt = [ repmat( '|c', 1, dicts_size + 1 ) '|'];
    hdr_frmt = ['\\rowcolor{black} \\color{white} \\textbf{Class} ' repmat( '& \\color{white} \\textbf{%d} ',  1, dicts_size ) '\\\\'];
    res_frmt = ['%s ' repmat( '& %f ',  1, dicts_size ) '\\\\'];
    
    
    disp('\begin{table}[H]')
    disp('\centering')
    disp(sprintf( '\\begin{tabular}{%s}', col_frmt ))
    disp('\hline')
    %disp('\rowcolor{black} \color{white} \textbf{Class} & \color{white} \textbf{AUC} \\')
    disp(sprintf(hdr_frmt, param_names))
    disp('\hline')
    
    for i=1:length(classes)
        
        disp( sprintf(res_frmt, classes{i}, auc(i,:)) )
        
    end
    
    disp('\hline')
    disp('\end{tabular}')
    disp(sprintf('\\caption{}'))
    disp(sprintf('\\label{tb:eval}'))
    disp('\end{table}')
end

