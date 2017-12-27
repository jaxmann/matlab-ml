function gradient_descent_OLS()
    
    filename = 'MLR.csv';
    A = csvread(filename);
    X = A(:,1:end-1); %all but last column%
    Xt = transpose(X); %transpose of X%
    y = A(:,end); % last column
    
    %least squares estimator is: beta=(XtX)^-1 * Xt y, as before%
        
    Bhat = inv(Xt*X) * Xt * y;
    
    filename2 = 'True_Beta.csv';
    Bstar = csvread(filename2);
   
    F = (1/30)*Xt*X;
    maxEigenvalue = eigs(F,1); %find largest eigenvalue of 1/n*Xt*X
    alpha = 1/maxEigenvalue; %stepsize is 1/L
        
    beta = zeros(30,1);
    m = length(y);
    betaLen = length(beta);
    h = beta; %copy of beta
    
    FbAtIterk = zeros(1000,1); %value of objective function at iteration k
    bHatNormAtk = zeros(1000,1); %norm(b at k - bhat)^2 at iteration k
    bStarNormAtk = zeros(1000,1); %norm(b at k - bstar)^2 at iteration k
    
    for iter = 1:1000 %1000 iterations of gradient descent
                
        FbAtIterk(iter,1) = objFunctionAt(y, X, beta);
        bHatNormAtk(iter, 1) = (norm(beta - Bhat))^2;
        bStarNormAtk(iter, 1) = (norm(beta - transpose(Bstar))^2);
        
        %temp is equivalent to difference b/w y(i) and h(x(i)) in gradient
        %descent update rule
        temp = (y - X*beta);
        
        %update beta with new values of hypothesis at x
        for i = 1:betaLen
           h(i,1) = sum(temp.*X(:,i));
        end
        
        beta = beta + (alpha/m)*h;
                
    end
        
    %plots
    
    FbetaHat = objFunctionAt(y, X, Bhat);
    plot(1:1000, log(FbAtIterk - FbetaHat));
    xlabel('Iterations');
    ylabel('log(f(b(k) - f(bhat))');
    title('log(f(b(k) - f(bhat)) vs Iterations');
    pause
       
    plot(1:1000, bStarNormAtk);
    xlabel('Iterations');
    ylabel('norm(b(k) - bstar)^2');
    title('norm(b(k) - bstar)^2 vs Iterations');
    pause
   
    plot(1:1000, bHatNormAtk);
    xlabel('Iterations');
    ylabel('norm(b(k) - bhat)^2');
    title('norm(b(k) - bhat)^2 vs Iterations');
    pause
    
    %d) While both figures for question b and c descend at a similar rate
    %(decreasing squared difference as iterations increase), chart c has a lower y
    %intercept than chart b implying that the difference between beta at k
    %and beta hat is smaller than the difference between beta at k and beta
    %star (for all k). I.e. norm(Bhat)^2) < norm(Bstar)^2, so OLS appears to
    %underestimate the magnitude of the betas. Note that only after around 
    %60000 iterations the squared difference become sclose (within 1%) of the true beta. 
   

end

%define objective separately so it can be called once for each iteration
%easily
function o = objFunctionAt(y, X, B)
    o = ((1/2)*(length(y)))*norm(y-X*B)^2;
end

