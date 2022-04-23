function[grad] = f_deriv(y,ybar,W,Wbar,u)

grad = zeros([u,1]);
sum1 = 0;
sum2 = 0;

for j=1:u

    sum1 = sum(W(:,j).*y(j,1)-W(:,j).*ybar);
    sum2 = sum(Wbar(:,j).*y(j,1)-Wbar(:,j).*y); 
    grad(j,1) = sum1 + sum2;

end