function   latex_table( classes, auc )
%LATEX_TABLE Summary of this function goes here
%   Detailed explanation goes here


    disp('\begin{table}[h!]')
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
    disp(sprintf('\\caption{AUC per Class}'))
    disp(sprintf('\\label{tb:eval}'))
    disp('\end{table}')
end

