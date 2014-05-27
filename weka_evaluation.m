function [ correct, prob ] = weka_evaluation( classifier, vecTest, gtTest )
%WEKA_EVALUATION Summary of this function goes here
%   Detailed explanation goes here

    % add weka path
%     javaaddpath('/home/evargasv/weka-3-7-11/weka.jar');
if strncmp(computer,'PC',2)
    javaaddpath('C:\Program Files\Weka-3-7\weka.jar');
elseif strncmp(computer,'GLNX',4)
    javaaddpath('/home/evargasv/weka-3-7-11/weka.jar');
elseif strncmp(computer,'MACI64',3)
    javaaddpath('/Applications/weka-3-7-11-apple-jvm.app/Contents/Resources/Java/weka.jar')
end

    import weka.*;
    
    % create feature name
    for i=1:size(vecTest,2)+1,
        f{i}=num2str(i);
    end
    
    % convert the features vector to an cell-array and add at the end of
    % each row, the label of the class to which it belongs
    for i=1:size(vecTest,1),
        
        for j=1:size(vecTest,2),
            ctest{i,j} = vecTest(i,j);
        end
        
        % add label of the class
        if( gtTest(i) > 0 ); label='pp'; else label='nn'; end;  
        ctest{i,j+1} = label;
    end
    
    % convert the cell-array to weka dataset
    wekaTest = convertWekaDataset('testing',f,ctest);
    
    %% testing with final confusion matrix
    cm = zeros(wekaTest.numClasses,wekaTest.numClasses);
    mx = zeros(1,wekaTest.numInstances);
    prob = zeros(1,wekaTest.numInstances);
    pmx = zeros(1,wekaTest.numInstances);
    predicted = zeros(wekaTest.numInstances,wekaTest.numClasses);
    for i=1:wekaTest.numInstances
        instance = wekaTest.instance(i-1);
        predicted = classifier.distributionForInstance(instance)';
        % maximum probability and maximum class(pmx)
        [mx(i),pmx(i)] = max(predicted);
        % probability of belonging to the class
        prob(i) = predicted(2);
        cm(instance.classValue+1,pmx(i)) = cm(instance.classValue+1,pmx(i))+1;
    end
    cm
    correct = sum(diag(cm))/sum(cm(:))

end

