rng('default');  % Random seed for reproducibility

n = 1000; % numero tot di sample
l = 20; % numero di sample etichettati
u = 980; % numero di sample non etichettati
nc = 500; %numero di sample per classe
lc = 10; %numero di sample etichettati per classe
uc = 490; %numero di sample non eichettati per classe

% sample generator with gaussian distribution
X = [gallery('normaldata',[nc 2],120)-2; ...
    gallery('normaldata',[nc 2],120)+4];
y = [ones(nc,1);-1*(ones(nc,1))];

% visualize the data
scatter(X(:,1),X(:,2),5,y,'filled')
X_samp=[X(1:lc,1:2); X(nc+1:nc+lc,1:2)];
y_samp=[y(1:lc); y(nc+1:nc+lc)];
scatter(X(:,1),X(:,2),5,'filled')
hold on 
scatter(X_samp(:,1),X_samp(:,2),5,y_samp,'filled')
hold off

% Estrazione dati non etichettati
X_unlabeld = [X(lc+1:nc,1:2);X(nc+lc+1:n,1:2)];

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
lambda_min = min(eig(W));

W_l = zeros([1,u]);
W_u = zeros([1,u]);

for i=1:u
    W_l = sum(W_samp(:,i));
    W_u = sum(W(:,i));
end
L = max(W_l+W_u)-lambda_min;

% fixed step
alpha = 1/L;

%Minimizziamo la loss function
y0 = -1 + 2.*rand(u,1); %cioè delle etichette casuali per i dati no-lab 
%y0 =sign( -1 + 2.*rand(u,1) ) per avere un label "realistico"
eps = 1e-4; %tolleranza
maxit = 100000; %iterazioni max

% metodo del gradiente con stepsize fisso
y_new = GD_fixed(alpha,y0,maxit,eps,y_samp,W,W_samp,u);  



