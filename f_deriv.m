function[grad] = f_deriv(y,y_samp,W,W_samp)

u=len(y);
grad = zeros([u,1]);
sum1 = 0;
sum2 = 0;

for j=1:u

    sum1 = sum( (W_samp(:,j)-W_samp(:,j)).*y_samp ) ;
    sum2 = sum( W(:,j).*y(j,1)-W(:,j).*y); 
    grad(j,1) = sum1 + sum2;

end
