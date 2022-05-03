rng('default');  % Random seed for reproducibility

n = 1000; % numero di punti  
p = 0.02; % frazione di punti etichettati

l = n*p; % numero di sample etichettati (occhio fallo intero)
samp=randperm(n,l);
indices=1:n;

u = 980; % numero di sample non etichettati
nc = 500; %numero di sample per classe
lc = 10; %numero di sample etichettati per classe
uc = 490; %numero di sample non eichettati per classe

% sample generator with gaussian distribution
X = [gallery('normaldata',[n/2 2],120)-2; gallery('normaldata',[n/2 2],120)+4];
y_tot = [ones(n/2,1);-1*(ones(n/2,1))];





% visualize the data
scatter(X(:,1),X(:,2),5,y_tot,'filled')
X_samp=X(samp);
y_samp=y_tot(samp);
%X_samp=[X(1:lc,1:2); X(nc+1:nc+lc,1:2)];
%y_samp=[y_tot(1:lc); y_tot(nc+1:nc+lc)];

y_exact=y_tot(setdiff(1:end,samp));


%y_exact=[y_tot(lc+1:nc);y_tot(nc+lc+1,n)];
scatter(X(:,1),X(:,2),5,'filled')
hold on 
%scatter(X_samp(:,1),X_samp(:,2),5,y_samp,'filled')
hold off



% Estrazione dati non etichettati
%X_unlabeld = [X(lc+1:nc,1:2);X(nc+lc+1:n,1:2)];
X_unlabeld = x(setdiff(1:end,samp));

%Calcolo distanze "utili", lab-nolab e nolab-lab
D_samp=zeros(l,u);
for i=1:l
    for j=1:u
        D_samp(i,j)=norm(X_samp(i,:)-X_unlabeld(j,:));
    end
end

D=zeros(u,u);
for i=1:u
    for j=1:u
        D(i,j)=norm(X_unlabeld(i,:)-X_unlabeld(j,:));
        D(j,i)=D(i,j);
    end
end

% Calcolo pesi exp(-x)
W_samp= exp(-D_samp);
W = exp(-D);

% Calcolo L, lipschitz constant

% lambda_min = min(eig(W));
lambda_max = max(eig(W));
W_l = zeros([1,u]);
W_u = zeros([1,u]);

for i=1:u
    W_l = sum(W_samp(:,i));
    W_u = sum(W(:,i));
end
%L = max(W_l+W_u)-lambda_min questa in realtà funziona ugualmente
L=sqrt(max(W_l+W_u)^2+lambda_max^2);


%Parameters for the gradient methods
y0 = -1 + 2.*rand(u,1); %cioè delle etichette casuali per i dati no-lab 
eps = 1e-4; %tolleranza
maxit = 1000; %iterazioni max

%che metodo vogliamo?
metodo="BCGDrand";
step_size=3; % regola per lo step_size
alpha=1/L;   % per lo step_size fisso
delta=0.5;   % per l'armijo rule

switch metodo
    case "GD"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%METODO DEL GRADIENTE CLASSICO

[y_GD_t, timeVec, Norms_t, accuracy]= ...
    GD_fixed(alpha,maxit,eps,y0,y_samp,W,W_samp,step_size,y_exact,delta); 

    case "BCGDcyc"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BCGD with cyclic rule

[y, timeVec, Norms, accuracy] = ...
    BCGD_cyclic_fixed(alpha,y0,maxit,eps,y_samp,W,W_samp,step_size,y_exact,delta);

    case "BCGDrand"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BCGD with randomized rule

[y, timeVec, Norms, accuracy] = BCGD_rand_fixed_uniform(alpha,maxit,eps,y0,y_samp,W,W_samp,step_size,y_exact,delta);

end



figure(1)
semilogy(Norms)
title('Norms Plot')
xlabel('iterations')
ylabel("Gradiet's norm")
figure(2)
plot(timeVec)
title('Time Plot')
xlabel('iterations')
ylabel('comulative time')
figure(3)
plot(accuracy)
title('Accuracy Plot')
xlabel('iterations')
ylabel('accuracy')

















