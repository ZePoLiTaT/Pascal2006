function   latex_table( classes, auc, dict_size )
%LATEX_TABLE Summary of this function goes here
%   Detailed explanation goes here


    disp('\begin{table}[H]')
    disp('\centering')
    disp('\begin{tabular}{|c|c|}')
    disp('\hline')
    disp('\rowcolor{black} \color{white} \textbf{Class} & \color{white} \textbf{AUC} \\')
    disp('\hline')
    
    for i=1:length(classes)
        disp(sprintf('%s & %f \\\\',classes{i}, auc(i) ))
    end
    
    disp('\hline')
    disp('\end{tabular}')
    disp(sprintf('\\Dictionary Size: %d words}', dict_size))
    disp(sprintf('\\label{tb:eval}'))
    disp('\end{table}')
end

