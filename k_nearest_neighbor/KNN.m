
%%%%%%%%%%%%%%%%%%%%%%PLOT OF TRAINING DATA


data= load('train-knn.mat');


A1 = data.a(:,3) == 1;
A2 = data.a(:,3) == 2;
A3 = data.a(:,3) == 3;

os = data.a(A1,:);
pluses = data.a(A2,:);
stars = data.a(A3,:);

% 
hold all
scatter(os(:,1), os(:,2)); %classification 1
scatter(pluses(:,1), pluses(:,2),'+'); %classification 2
scatter(stars(:,1), stars(:,2),'*'); %classification 3
ylim([-2 11]);
xlim([-2 11]);
xlabel('x');
ylabel('y');
title('knn training data');
pause

% ---------------------------
% minX = min(data.a(:,1));
% maxX = max(data.a(:,1));
% minY = min(data.a(:,2));
% maxY = max(data.a(:,2));
% 
% rng(2);
% kCentroids = zeros(NUM_K, 2);
% 
% hold all
% scatter(data.a(:,1), data.a(:,2));
% for i = 1:NUM_K
%     rX = (maxX-minX).*rand + minX;
%     rY = (maxY-minY).*rand + minY;
%     kCentroids(i, 1) = rX;
%     kCentroids(i,2) = rY;
%     scatter(kCentroids(i,1), kCentroids(i,2));
% end
% 
% xlabel('x');
% ylabel('y');
% title('knn');
% pause
% -----------------------------


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%% TESTING DATA - CLASSIFICATION RESULTS SAVED IN
%%%%%%%%%%%%%% test_label_l1/l2


testdata = load('test-knn.mat');


K = 20;

test_label_l1 = zeros([40000 1]);
test_label_l2 = zeros([40000 1]);

for m = 1:length(testdata.B)
    l1dist = zeros(150,2);
    l2dist = zeros(150,2);
    for j = 1:150
        l1dist(j,1) = abs( testdata.B(m,1) - data.a(j,1) ) + abs( testdata.B(m,2) - data.a(j,2) ); %l1
        l2dist(j,1) = sqrt( (testdata.B(m,1) - data.a(j,1) )^2 + ( testdata.B(m,2) - data.a(j,2) )^2 ); %l2
    end

    l1dist(:,2) = data.a(:,3); %acppend classification to second col
    l2dist(:,2) = data.a(:,3); %append classifcation to second col
    % pick distances that are smallest, and select most common
    % classification among those

    sortedl1dist = sortrows(l1dist,1);
    sortedl2dist = sortrows(l2dist,1);

    kSmallestEltsl1 = sortedl1dist(1:K,:); % k smallest are the k closest after being sorted
    kSmallestEltsl2 = sortedl2dist(1:K,:);

    % will assign new element to MOST COMMON classification within nearest k
    % elements

    modeOfClassificationsl1 = mode(kSmallestEltsl1(:,2));
    modeOfClassificationsl2 = mode(kSmallestEltsl2(:,2));


    test_label_l1(m) = modeOfClassificationsl1;
    test_label_l2(m) = modeOfClassificationsl2;


end

% disp(test_label_l1);
% disp(test_label_l2);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%% CLASSIFICATION RESULTS

fullL1Data = zeros(40000,3);
fullL1Data(:,1:2) = testdata.B;
fullL1Data(:,3) = test_label_l1;

l1A1 = fullL1Data(:,3) == 1;
l1A2 = fullL1Data(:,3) == 2;
l1A3 = fullL1Data(:,3) == 3;

l1os = fullL1Data(l1A1,:);
l1pluses = fullL1Data(l1A2,:);
l1stars = fullL1Data(l1A3,:);

hold all
scatter(l1os(:,1), l1os(:,2));
scatter(l1pluses(:,1), l1pluses(:,2),'+');
scatter(l1stars(:,1), l1stars(:,2),'*');
ylim([0 7.8])
xlim([0 7.8])
xlabel('x');
ylabel('y');
title(strcat('knn L1 k=', num2str(K)));
pause

fullL2Data = zeros(40000,3);
fullL2Data(:,1:2) = testdata.B;
fullL2Data(:,3) = test_label_l2;

l2A1 = fullL2Data(:,3) == 1;
l2A2 = fullL2Data(:,3) == 2;
l2A3 = fullL2Data(:,3) == 3;

l2os = fullL2Data(l2A1,:);
l2pluses = fullL2Data(l2A2,:);
l2stars = fullL2Data(l2A3,:);
% 
hold all
scatter(l2os(:,1), l2os(:,2));
scatter(l2pluses(:,1), l2pluses(:,2),'+');
scatter(l2stars(:,1), l2stars(:,2),'*');
ylim([0 7.8])
xlim([0 7.8])
xlabel('x');
ylabel('y');
title(strcat('knn L2 k=',num2str(K)));
pause



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


