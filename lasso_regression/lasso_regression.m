load('train-greedy.mat')
load('valid-greedy.mat')
load('test-greedy.mat')
load('true-beta.mat')


lasso_wrapper(train,validation,test,beta);


%Input: train data, validation data and test data.
%Output: optimal beta, optimal lambda, estimation error and prediction
%        error. 
function [bestBeta] = lasso_wrapper(train,validation,test,beta)

%ridge regression normally solves the B that minimizes (||y - XB|| + L||B||
%but with intercept term it should be the B that minimizes
% ||y - B0 - XB|| + L||B|| (i think) - so find lambda that minmizes this for
% training data

d = size(train, 2) - 1; %"d"
X = train(:,1:end-1);
y = train(:,end);

dvalidation = size(validation, 2) - 1; 
Xvalidation = validation(:,1:end-1);
yvalidation = validation(:,end);


[B, FitInfo] = lasso(X,y); %TEST x and y

% disp(FitInfo)
% disp(FitInfo.Lambda)

minLambda = FitInfo.Lambda(1);
minIndex = 1;

%initialize minFuncVal to first lambda value (USING VALIDATION SET)
minFuncVal = norm(yvalidation - FitInfo.Intercept(1)*ones(100,1) - Xvalidation*B)^2;

for i = 1:length(FitInfo.Lambda)
   newval = norm(yvalidation - FitInfo.Intercept(i)*ones(100,1) - Xvalidation*B)^2;
   if newval < minFuncVal
      minLambda = FitInfo.Lambda(i);
      minIndex = i;
      minFuncVal = newval;
   end
end

disp(minLambda)
disp(minIndex)

bestBeta = B(:,minIndex); %select bestBeta to be the column for which the lambda was lowest

est_error = norm(bestBeta - beta);
%take intercept of corresponding best lambda found (i.e. find index of best
%lambda and pick that intercept)
pred_error = (1/size(test,1)) * norm(test(:,end) - FitInfo.Intercept(minIndex)*ones(size(test,1),1) - test(:,1:end-1)*bestBeta) ^ 2;


disp('estimation error') 
disp(est_error);

disp('prediction error')
disp(pred_error);

end


