load('train-greedy.mat')
load('valid-greedy.mat')
load('test-greedy.mat')
load('true-beta.mat')


ridge_reg(train,validation,test,beta);


%Input: train data, validation data and test data
%Output: optimal beta, optimal lambda, estimation error and prediction error.
%        Plot your errors as iteration changes
function [ridgeBeta, optimal_lambda, estimation_error, prediction_error] = ridge_reg(train,validation,test,beta)


%only minimize first term for beta?

d = size(train, 2) - 1; %"d"
X = train(:,1:end-1);
y = train(:,end);

dvalidation = size(validation, 2) - 1; 
Xvalidation = validation(:,1:end-1);
yvalidation = validation(:,end);


lambdas = [0.0125 0.025 0.05 0.1 0.2];
estErrAtLambda = [];
predErrAtLambda = [];
    

% ridgeBeta = ridge(y,X,lambdas);
% disp(size(ridgeBeta));

%this takes ages to run the optimizer... beware (like 1 hr+)
for y = 1:length(lambdas)
    

    func = @(beta) (1/(2*size(test,1))) * norm(y - X*beta)^2 + lambdas(y)* norm(beta)^2;
    beta0 = zeros(d, 1);
    options = optimset('MaxFunEvals',100000, 'MaxIter',100000);
    ridgeBeta = fminsearch(func, beta0, options);
% 
%     %disp(ridgeBeta);
%     

    est_error = norm(ridgeBeta - beta);
    pred_error =  norm(validation(:,end) - validation(:,1:end-1)*ridgeBeta) ^ 2;
%     
    estErrAtLambda = [estErrAtLambda est_error];
    predErrAtLambda = [predErrAtLambda pred_error];
    
%     disp('lambda is')
%     disp(lambdas(y))
% 
%     disp('estimation error') 
%     disp(est_error);
% 
%     disp('prediction error')
%     disp(pred_error);
% 
%     disp('------')
   
    %run these 4 lines along with the ridgeBeta = ridge(y,X,k) line above
    %to run the matlab ridge version
%     est_error = norm(ridgeBeta(:,y) - beta);
%     pred_error =  (1/size(validation,1)) * norm(validation(:,end) - validation(:,1:end-1)*ridgeBeta(:,y)) ^ 2;
%     
%     estErrAtLambda = [estErrAtLambda est_error];
%     predErrAtLambda = [predErrAtLambda pred_error];
    

end

plot(lambdas, estErrAtLambda);
xlabel('lambda');
ylabel('estimation error');
title('estimation error vs lambda');
pause

plot(lambdas, predErrAtLambda);
xlabel('lambda');
ylabel('prediction error');
title('prediction error vs lambda');
pause


end
