function ordinary_least_squares()
    filename = 'MLR.csv';
    A = csvread(filename);
    X = A(:,1:end-1); %all but last column%
    Xt = transpose(X); %transpose of X%
    y = A(:,end); % last column i.e y
    
    %least squares estimator is: beta=(XtX)^-1 * Xt y%
    
    Bhat = inv(Xt*X) * Xt * y;
        
    filename2 = 'True_Beta.csv';
    Bstar = csvread(filename2);
        
    squaredError = norm(Bhat - transpose(Bstar)).^2;
    
    disp('squared error is:');
    disp(squaredError)
    
end

