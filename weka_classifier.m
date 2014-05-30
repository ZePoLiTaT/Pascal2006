function [ classifier ] = weka_classifier( vecTrain, gtTrain )
%WEKA_CLASSIFIER Summary of this function goes here
%
%   - vecTrain: vector of dimension N by d, containing N features of size d 
%   - gtTrain: vector of dimension N by 1, containing 1 if the element
%   belongs to the class and -1 otherwise

    % add weka path
if strncmp(computer,'PC',2)
    javaaddpath('C:\Program Files\Weka-3-7\weka.jar');
elseif strncmp(computer,'GLNX',4)
    javaaddpath('/home/evargasv/weka-3-7-11/weka.jar');
elseif strncmp(computer,'MACI64',3)
    javaaddpath('/Applications/weka-3-7-11-apple-jvm.app/Contents/Resources/Java/weka.jar')
end

    import weka.*;
    
    % create feature name
    for i=1:size(vecTrain,2)+1,
        f{i}=num2str(i);
    end
    
    % convert the features vector to an cell-array and add at the end of
    % each row, the label of the class to which it belongs
    for i=1:size(vecTrain,1),
        for j=1:size(vecTrain,2),
            ctrain{i,j} = vecTrain(i,j);
        end
        % add label of the class
        if( gtTrain(i) > 0 ); label='pp'; else label='nn'; end;  
        ctrain{i,j+1} = label;
    end
    
    % convert the cell-array to weka dataset
    wekaTrain = convertWekaDataset('training',f,ctrain);
    
    % create classifier instance and train it

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Train the classifier

    %Naive Bayes
    %classifier = trainWekaClassifier(wekaTrain,'bayes.NaiveBayes');

    %SVM
    %classifier = trainWekaClassifier(wekaTrain,'functions.SMO');

    % AdaBoost
    classifier = trainWekaClassifier(wekaTrain,'meta.AdaBoostM1');

    %RandomForest
    %classifier = trainWekaClassifier(wekaTrain,'trees.RandomForest');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

