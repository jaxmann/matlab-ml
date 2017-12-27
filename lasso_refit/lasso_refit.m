%In the train, valid and test data, remember the last column is label. Plot
%your figures in the defination of functions.
load('train-greedy.mat')
load('valid-greedy.mat')
load('test-greedy.mat')
load('true-beta.mat')


refined_est(train,validation,test,beta);

%Input: train data, validation data and test data.
%Output: optimal beta, optimal lambda, estimation error and prediction
%        error. 
function [bestBeta] = lasso_wrapper(train,validation,test,beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%         Your Code Starts Here         %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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



%refined optimizer.
%Output: refined beta and estimation error.
function refined_est(train,validation,test,beta)


d = size(train, 2) - 1; %"d"
X = train(:,1:end-1);
y = train(:,end);


%find indices where BetaLasso != 0;

lassoBeta = lasso_wrapper(train, validation, test, beta);

indicesWhereLassoBetaNot0 = [];

for i = 1:length(lassoBeta)
   if lassoBeta(i) ~= 0
      indicesWhereLassoBetaNot0 = [indicesWhereLassoBetaNot0 i]; %set of all indices 
   end
end

inds = indicesWhereLassoBetaNot0;

subX = zeros(size(X,1), length(inds));
for s = 1:length(inds)
   subX(:,s) = X(:, inds(s)); 
end

%sub X is the subset of X columns for which the corresponding Bj = 0 -so
%when we optimize for the coefficient beta we only give it a chance to
%optimize on the columns for which Bj != 0 (and then reconstruct the entire
%beta again later with the correct indices handled etc);

func = @(beta) norm(y - subX*beta)^2;
beta0 = zeros(length(inds), 1);
options = optimset('MaxFunEvals',1000000, 'MaxIter',1000000);
betaMin = fminsearch(func, beta0, options);


fullBetaMinRefit = zeros(d,1);
for a = 1:length(inds)
    fullBetaMinRefit(inds(a),1) = betaMin(a);
end


est_error_lasso = norm(lassoBeta - beta);

disp('estimation error of lasso') 
disp(est_error_lasso);


est_error_refit = norm(fullBetaMinRefit - beta);

disp('estimation error of refit') 
disp(est_error_refit); %overfitting?


end

