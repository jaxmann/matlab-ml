%In the train, valid and test data, remember the last column is label. Plot
%your figures in the defination of functions.
load('train-greedy.mat')
load('valid-greedy.mat')
load('test-greedy.mat')
load('true-beta.mat')


forward_greedy(train,validation,test,beta);

%forward greedy algorithm
%Input: train data, validation data and test data
%Output: number of optimized features, optimal beta, estimation error and 
%        prediction error.
%        Plot your errors as iteration changes
function [numFeatures, optimalBeta, estimation_error, prediction_error] = forward_greedy(train,validation,test,beta)

dvalidation = size(validation, 2) - 1; %validation dimension
Xvalidation = validation(:,1:end-1); %validation X (not including last col which is y)
yvalidation = validation(:,end); %validation y


%as above, but for training data
d = size(train, 2) - 1; %"d"
X = train(:,1:end-1); 
y = train(:,end);


%figure out K by using the validation dataset
iterations = 8; %i.e K

%intialize
A = [];
B = zeros(d, iterations + 1); %first col remains all 0 due to initialization
%first col of B is just 0s since we have a B(0) initialized to 0
%intentionally
numFeatures = 0;
optimalBeta = zeros(d,1);

estimation_error_at_k = zeros(iterations + 1, 1); %have to mess with indices a bit b/c matlab 1 indexed instead of 0... 
prediction_error_at_k = zeros(iterations + 1, 1);

%previous validation error -if curr is greater than previous, break (see
%below)
prevValidationError = norm(yvalidation - Xvalidation * optimalBeta)^2;

actualK = 0; %k after having broken the loop early (set below)

%greedy algorithm
for k = 2:iterations + 1
    imax = 0;
    largestIndex = 0;
    for ii = 1:d
        newval = abs(transpose(X(:,ii))*(X*B(:, k-1) - y));
        if newval > imax
            imax = newval;
            largestIndex = ii;
        end        
    end
    %
    A = union(A, [largestIndex]); %[A largestIndex]; % append largest ii (i.e. optimal feature) to feature list A
   
    %find beta that minimizes y-xB
    
    sortedA = sort(A);
    subX = zeros(size(X,1), length(sortedA));
    for s = 1:length(sortedA)
       subX(:,s) = X(:, sortedA(s)); 
    end
    
    func = @(beta)norm(y - subX*beta);
    beta0 = zeros(length(sortedA), 1);
    options = optimset('MaxFunEvals',50000, 'MaxIter',100000);
    betaMin = fminsearch(func, beta0,options);
    
    fullBetaMin = zeros(d,1);
    for a = 1:length(sortedA)
        fullBetaMin(sortedA(a),1) = betaMin(a);
    end
    
    optimalBeta = fullBetaMin;
    
    %check validation error - once validation errors starts increasing,
    %break out of the loop and select that number of iterations as the
    %optimal
    currValidationError = norm(yvalidation - Xvalidation * optimalBeta)^2;
    
    if currValidationError <= prevValidationError
        prevValidationError = currValidationError;
    elseif currValidationError > prevValidationError
        actualK = k;   
        break;
    end

%     actualK = k;
   
    
%     disp(k)
%     disp(A)
%     disp(optimalBeta(end-2:end, 1))

    
    B(:,k) = fullBetaMin;
    
    numFeatures = size(A,2);
    
    
    
    est_error = norm(optimalBeta -beta);
    pred_error = 1/size(test,1) * norm(test(:,end) - test(:,1:end-1)*optimalBeta) ^ 2;
    
    estimation_error_at_k(k,1) = est_error;
    prediction_error_at_k(k,1) = pred_error;
    
%     disp(k)
%    
%     disp('estimation error') 
%     disp(est_error);
%     
%     disp('prediction error')
%     disp(pred_error);
    
end

%optimal beta at end is the optimal beta at the highest K which is also the
%lowest error beta (the best one)

estimation_error = estimation_error_at_k(actualK-1, 1);
prediction_error = prediction_error_at_k(actualK-1, 1);

plot(1:actualK-1, estimation_error_at_k(2:actualK));
xlabel('K');
xticks(1:actualK)
ylabel('estimation error');
title('estimation error vs K');
pause

plot(1:actualK-1, prediction_error_at_k(2:actualK));
xlabel('K');
xticks(1:actualK)
ylabel('prediction error');
title('prediction error vs K');
pause

%disp('number of optimized features is')
%disp(numFeatures);

% disp('optimal beta is')
% disp(optimalBeta)

% disp('ending estimation error')
% est_error = norm(optimalBeta -beta);
% disp(est_error);
% 
% disp('ending prediction error')
% pred_error = 1/size(test,1) * norm(test(:,end) - test(:,1:end-1)*optimalBeta) ^ 2;
% disp(pred_error);

    
end

