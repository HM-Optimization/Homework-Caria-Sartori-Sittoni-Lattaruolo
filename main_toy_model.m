rng('default');  % Random seed for reproducibility

n = 10000; % number of points
p = 0.02; % fraction of labeled points

% sample generator with gaussian distribution
l = n*p; % number of labeled samples
u = n*(1-p);  % number of unlabeled samples
samp=randperm(n,l); % random sample the labeled indices
indices=1:n;
X = [gallery('normaldata',[n/2 2],120)-2; gallery('normaldata',[n/2 2],120)+4];
y_tot = [ones(n/2,1);-1*(ones(n/2,1))];

% build the data
X_samp=X(samp,:);
X_unlabeled = X(setdiff(1:end,samp),:);
y_samp=y_tot(samp);
y_exact=y_tot(setdiff(1:end,samp));

%visualize the data
figure(1)
scatter(X(:,1),X(:,2),5,y_tot,'filled')  % plot with true labels
figure(2);
scatter(X(:,1),X(:,2),5,'filled')
hold on 
scatter(X_samp(:,1),X_samp(:,2),25,y_samp,'filled')  % plot with the small set of labeled data
hold off 


% Compute the distances between lab-nolab and nolab-nolab
D_samp=pdist2(X_samp,X_unlabeled);
D=pdist2(X_unlabeled,X_unlabeled);
% Calcolate the weights exp(-dist)
W_samp= exp(-D_samp);
W = exp(-D);

% Parameters for the gradient methods
y0 = -1 + 2.*rand(u,1); % starting points
eps = 1e-4; % tollerance
maxit = 1000; % max iteration

disp("fixed step size:1 / armijo rule:2 / exact line search:3")
step_size=input('Choose step size rule:'); % step_size update rule
delta=0.5;   % parameter for the armijo rule

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CLASSIC GRADIENT METHOD 
    
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


%PLOT THE RESULTS
figure(3)
hold on
semilogy(Norms_GD,LineWidth = 2)
semilogy(Norms_BCGDcyc,':',LineWidth = 2)
semilogy(Norms_BCGDrand,'--',LineWidth = 2)
hold off
title('Norms Plot')
legend('GD','BCGDcyc','BCGDrand')
xlabel('iterations')
ylabel("Gradiet's norm")

figure(4)
hold on
plot(timeVec_GD,accuracy_GD,LineWidth = 2)
plot(timeVec_BCGDcyc,accuracy_BCGDcyc,':',LineWidth = 2)
plot(timeVec_BCGDrand,accuracy_BCGDrand,'--',LineWidth = 2)
hold off
title('Accuracy vs Time')
legend('GD','BCGDcyc','BCGDrand')
xlabel('time')
ylabel('accuracy')

figure(5)
hold on
plot(accuracy_GD,LineWidth = 2)
plot(accuracy_BCGDcyc,':',LineWidth = 2)
plot(accuracy_BCGDrand,'--',LineWidth = 2)
hold off
title('Accuracy Plot')
legend('GD','BCGDcyc','BCGDrand')
xlabel('iterations')
ylabel('accuracy')

%visualize final data labeled
figure(6)
scatter(X_unlabeled(:,1),X_unlabeled(:,2),5,y_GD,'filled')
figure(7)
scatter(X_unlabeled(:,1),X_unlabeled(:,2),5,y_BCGDcyc,'filled')
figure(8)
scatter(X_unlabeled(:,1),X_unlabeled(:,2),5,y_BCGDrand,'filled')
