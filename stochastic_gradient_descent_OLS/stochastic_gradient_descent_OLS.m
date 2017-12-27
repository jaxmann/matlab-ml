function stochastic_gradient_descent_OLS()
    
    filename = 'MLR.csv';
    A = csvread(filename);
    X = A(:,1:end-1); %all but last column%
    Xt = transpose(X); %transpose of X%
    y = A(:,end); % last column
    
    %least squares estimator is: beta=(XtX)^-1 * Xt y%
        
    Bhat = inv(Xt*X) * Xt * y;
    
    filename2 = 'True_Beta.csv';
    Bstar = csvread(filename2);
    
    F = (1/30)*Xt*X;
    maxEigenvalue = eigs(F,1); %max eigenvector of (1/n)*Xt*X as in P2
    
    beta = zeros(30,1);
    m = length(y);
    h = beta;
    
    %%%%%%%%%%%%%%%%%%%%%%% .1/L step size
    beta = zeros(30,1);
    alpha = .1/maxEigenvalue; %stepsize 0.1/L
    
    FbAtIterpass = zeros(20,1); %value of objective function at pass number x of 20
    bHatNormAtpass = zeros(20,1); %value of norm(bhat -b(k))^2
    bStarNormAtpass = zeros(20,1); %value of norm(bstar - b(k))^2
    
    for iter = 1:20
        
        FbAtIterpass(iter,1) = objFunctionAt(y, X, beta);
        bHatNormAtpass(iter, 1) = (norm(beta - Bhat))^2;
        bStarNormAtpass(iter, 1) = (norm(beta - transpose(Bstar))^2);
              
        for ii = 1:m
            for jj = 1:30 %update beta for each column, for each row, using OLS update function
                beta(jj) = beta(jj) + (alpha)*((y(ii) - sum(X(ii,:)*beta))*X(ii,jj));
            end
        end
        
    end
                
    FbetaHat = objFunctionAt(y, X, Bhat);
    plot(1:20, log(FbAtIterpass - FbetaHat));
    xlabel('Iterations');
    ylabel('log(f(b(k) - f(bhat))');
    title('log(f(b(k) - f(bhat)); stepsize=.1/L');
    pause
       
    plot(1:20, bStarNormAtpass);
    xlabel('Iterations');
    ylabel('norm(b(k) - bstar)^2');
    title('norm(b(k) - bstar)^2; stepsize=.1/L');
    pause
   
    plot(1:20, bHatNormAtpass);
    xlabel('Iterations');
    ylabel('norm(b(k) - bhat)^2');
    title('norm(b(k) - bhat)^2; stepsize=.1/L');
    pause
    
    %%%%%%%%%%%%%%%%%%%%%%% 1.7/L stepsize
    beta = zeros(30,1);
    alpha = 1.7/maxEigenvalue; %stepsize 1.7/L
    
    FbAtIterpass = zeros(20,1);
    bHatNormAtpass = zeros(20,1);
    bStarNormAtpass = zeros(20,1);
    
    for iter = 1:20
        
        FbAtIterpass(iter,1) = objFunctionAt(y, X, beta);
        bHatNormAtpass(iter, 1) = (norm(beta - Bhat))^2;
        bStarNormAtpass(iter, 1) = (norm(beta - transpose(Bstar))^2);
              
        for ii = 1:m
            for jj = 1:30
                beta(jj) = beta(jj) + (alpha)*((y(ii) - sum(X(ii,:)*beta))*X(ii,jj));
            end
        end
        
    end
                
    FbetaHat = objFunctionAt(y, X, Bhat);
    plot(1:20, log(FbAtIterpass - FbetaHat));
    xlabel('Iterations');
    ylabel('log(f(b(k) - f(bhat))');
    title('log(f(b(k) - f(bhat)); stepsize=1.7/L');
    pause
       
    plot(1:20, bStarNormAtpass);
    xlabel('Iterations');
    ylabel('norm(b(k) - bstar)^2');
    title('norm(b(k) - bstar)^2; stepsize=1.7/L');
    pause
   
    plot(1:20, bHatNormAtpass);
    xlabel('Iterations');
    ylabel('norm(b(k) - bhat)^2');
    title('norm(b(k) - bhat)^2; stepsize=1.7/L');
    pause
    
    
    %%%%%%%%%%%%%%%%%%% 1/L stepsize
    beta = zeros(30,1);
    alpha = 1/maxEigenvalue; %stepsize 1/L
    
    FbAtIterpass = zeros(20,1);
    bHatNormAtpass = zeros(20,1);
    bStarNormAtpass = zeros(20,1);
    
    for iter = 1:20
        
        FbAtIterpass(iter,1) = objFunctionAt(y, X, beta);
        bHatNormAtpass(iter, 1) = (norm(beta - Bhat))^2;
        bStarNormAtpass(iter, 1) = (norm(beta - transpose(Bstar))^2);
              
        for ii = 1:m
            for jj = 1:30
                beta(jj) = beta(jj) + (alpha)*((y(ii) - sum(X(ii,:)*beta))*X(ii,jj));
            end
        end
        
    end
                
    FbetaHat = objFunctionAt(y, X, Bhat);
    plot(1:20, log(FbAtIterpass - FbetaHat));
    xlabel('Iterations');
    ylabel('log(f(b(k) - f(bhat))');
    title('log(f(b(k) - f(bhat)); stepsize=1/L');
    pause
       
    plot(1:20, bStarNormAtpass);
    xlabel('Iterations');
    ylabel('norm(b(k) - bstar)^2');
    title('norm(b(k) - bstar)^2; stepsize=1/L');
    pause
   
    plot(1:20, bHatNormAtpass);
    xlabel('Iterations');
    ylabel('norm(b(k) - bhat)^2');
    title('norm(b(k) - bhat)^2; stepsize=1/L');
    pause
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% 0.01/L stepsize
    beta = zeros(30,1);
    alpha = 0.01/maxEigenvalue; %stepsize 0.01/L
    
    FbAtIterpass = zeros(20,1);
    bHatNormAtpass = zeros(20,1);
    bStarNormAtpass = zeros(20,1);
    
    for iter = 1:20
        
        FbAtIterpass(iter,1) = objFunctionAt(y, X, beta);
        bHatNormAtpass(iter, 1) = (norm(beta - Bhat))^2;
        bStarNormAtpass(iter, 1) = (norm(beta - transpose(Bstar))^2);
              
        for ii = 1:m
            for jj = 1:30
                beta(jj) = beta(jj) + (alpha)*((y(ii) - sum(X(ii,:)*beta))*X(ii,jj));
            end
        end
        
    end
                
    FbetaHat = objFunctionAt(y, X, Bhat);
    plot(1:20, log(FbAtIterpass - FbetaHat));
    xlabel('Iterations');
    ylabel('log(f(b(k) - f(bhat))');
    title('log(f(b(k) - f(bhat)); stepsize=0.01/L');
    pause
       
    plot(1:20, bStarNormAtpass);
    xlabel('Iterations');
    ylabel('norm(b(k) - bstar)^2');
    title('norm(b(k) - bstar)^2; stepsize=0.01/L');
    pause
   
    plot(1:20, bHatNormAtpass);
    xlabel('Iterations');
    ylabel('norm(b(k) - bhat)^2');
    title('norm(b(k) - bhat)^2; stepsize=0.01/L');
    pause

       
    
    %e) As step size increases, it takes fewer iterations for the stochastic
    %gradient descent algorithm to minimize the squared difference of the
    %betas, (although presumably this could cause some trouble such as
    %oscillating around the minimum). Note that the graphs only begin after
    %1 iteration has been completed.
   

end


%define objective function separately for ease of use, as in P2
function o = objFunctionAt(y, X, B)
    o = ((1/2)*(length(y)))*norm(y-X*B)^2;
end

