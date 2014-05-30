%Test weka's Naive Bayes implementation on the iris data. 

% add weka path
javaaddpath('/home/evargasv/weka-3-6-11/weka.jar');

load fisheriris;    %built in to matlab

%Shuffle the data
rand('twister',0);
perm = randperm(150);
meas = meas(perm,:);
species = species(perm,:);

featureNames = {'sepallength','sepalwidth','petallength','petalwidth','class'};

%Prepare test and training sets. 
data = [num2cell(meas),species];
train = data(1:120  ,:);
test  = data(121:end,:);

classindex = 5;

%Convert to weka format
train = matlab2weka('iris-train',featureNames,train,classindex);
test =  matlab2weka('iris-test',featureNames,test);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train the classifier

%Naive Bayes
%nb = trainWekaClassifier(train,'bayes.NaiveBayes');

%SVM
% nb = trainWekaClassifier(train,'functions.SMO');

% AdaBoost
%nb = trainWekaClassifier(train,'meta.AdaBoostM1');

%RandomForest
%nb = trainWekaClassifier(train,'trees.RandomForest');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Test the classifier
[predicted, classProbs] = wekaClassify(test,nb);

%The actual class labels (i.e. indices thereof)
actual = test.attributeToDoubleArray(classindex-1); %java indexes from 0

errorRate = sum(actual ~= predicted)/30

