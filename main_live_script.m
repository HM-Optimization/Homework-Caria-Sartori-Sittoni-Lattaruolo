rng('default');  % Random seed for reproducibility

% Load Occupancy Detection Dataset
opts = detectImportOptions('data_occupancy.txt',"VariableNamingRule","preserve");
% Feature selection
opts.SelectedVariableNames = {'Var3', 'Var5','Var8'};
data = readtable('data_occupancy.txt',opts);
data = rmmissing(data);

% Convertion from table to array matrix
X1 = table2array(data(:,1));
X2 = table2array(data(:,2));
Y = table2array(data(:,3));

% Changing class 0 to class -1
Y=2*Y-1

% Removing outliers
X1 = X1(X2<900);
X2 = X2(X2<900);
Y = Y(X2<900);

% Re-scaling the features
X1 = 100*(X1 - min(X1)) / (max(X1)-min(X1));
X2 = 100*(X2 - min(X2)) / (max(X2)-min(X2));

% Plot of the dataset
figure
scatter(X1,X2,5,Y,'filled');
title('Occupancy Detection Dataset');
xlabel('Temperature');
ylabel('Light');


n = length(Y); % number of points  
p = 0.02; % fraction of labeled points

% sample generator with gaussian distribution
l = ceil(n*p); % number of labeled samples
u=n-l; % number of unlabeled samples
samp=randperm(n,l); 
indices=1:n;
X = [X1,X2];
y_tot = Y;


% build the data
X_samp=X(samp,:);
X_unlabeled = X(setdiff(1:end,samp),:);
y_samp=y_tot(samp);
y_exact=y_tot(setdiff(1:end,samp));

% visualize the data
figure;
scatter(X(:,1),X(:,2),5,'filled')
hold on 
scatter(X_samp(:,1),X_samp(:,2),20,y_samp,'filled')
title('Occupancy Detection Dataset - 2% labeled')
xlabel('Temperature');
ylabel('Light');
hold off



% Compute the distances between lab-nolab and nolab-nolab
D_samp=pdist2(X_samp, X_unlabeled);

D=pdist2(X_unlabeled,X_unlabeled);

% Calcolate the weights exp(-dist)
W_samp= exp(-D_samp);
W = exp(-D);

% Parameters for the gradient methods
y0 = zeros(u,1); % starting points
eps = 1e-4; % tollerance
maxit = 1000; % max iteration

disp("fixed step size:1 / armijo rule:2 / exact line search:3")
step_size=input('Choose step size rule:'); % step_size update rule
delta=0.5;   % parameter for the armijo rule

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLASSIC GRADIENT METHOD 
    
[y_GD, timeVec_GD, Norms_GD, accuracy_GD]= ...
    GD(maxit,eps,y0,y_samp,W,W_samp,step_size,y_exact,delta); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BCGD with cyclic rule

[y_BCGDcyc, timeVec_BCGDcyc, Norms_BCGDcyc, accuracy_BCGDcyc] = ...
    BCGD_cyclic(y0,maxit,eps,y_samp,W,W_samp,step_size,y_exact,delta);

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BCGD with randomized rule

[y_BCGDrand, timeVec_BCGDrand, Norms_BCGDrand, accuracy_BCGDrand] = ...
    BCGD_random(maxit,eps,y0,y_samp,W,W_samp,step_size,y_exact,delta);

% PLOT THE RESULTS

% Norms plots
figure(3)
semilogy(Norms_GD,LineWidth = 2)
hold on
semilogy(Norms_BCGDcyc,':',LineWidth = 2)
semilogy(Norms_BCGDrand,'--',LineWidth = 2)
hold off
title('Occupancy Detection Dataset - Norms Plot')
xlabel('iterations')
ylabel("Gradiet's norm")
legend('GD', 'BCGDcyc', 'BCGDrand')

% Accuracy vs Time
figure(4)
plot(timeVec_GD,accuracy_GD,LineWidth = 2)
hold on
plot(timeVec_BCGDcyc,accuracy_BCGDcyc,':',LineWidth = 2)
plot(timeVec_BCGDrand,accuracy_BCGDrand,'--',LineWidth = 2)
hold off
title('Occupancy Detection Dataset - Accuracy vs Time')
xlabel('time')
ylabel('accuracy')
legend('GD', 'BCGDcyc', 'BCGDrand')

% Accuracy plots
figure(5)
plot(accuracy_GD,LineWidth = 2)
hold on
plot(accuracy_BCGDcyc,':',LineWidth = 2)
plot(accuracy_BCGDrand,'--',LineWidth = 2)
hold off
title('Occupancy Detection Dataset - Accuracy Plot')
xlabel('iterations')
ylabel('accuracy')
legend('GD', 'BCGDcyc', 'BCGDrand')


% VISUALIZE FINAL DATA LABELED

% Gradient Descent
figure(6)
scatter(X_unlabeled(:,1),X_unlabeled(:,2),5,sign(y_GD),'filled')
title('Occupancy Detection Dataset - Gradient Descent')
xlabel('Temperature');
ylabel('Light');

% BCGD cyclic
figure(7)
scatter(X_unlabeled(:,1),X_unlabeled(:,2),5,sign(y_BCGDcyc),'filled')
title('Occupancy Detection Dataset - BCGD cyclic')
xlabel('Temperature');
ylabel('Light');

% BCGD random
figure(8)
scatter(X_unlabeled(:,1),X_unlabeled(:,2),5,sign(y_BCGDrand),'filled')
title('Occupancy Detection Dataset - BCGD random')
xlabel('Temperature');
ylabel('Light');
