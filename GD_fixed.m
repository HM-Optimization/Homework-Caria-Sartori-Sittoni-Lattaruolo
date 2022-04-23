function[y] = GD_fixed(alpha,y0,maxit,eps,ybar,W,Wbar,u)
% Gradient descent per stepsize fisso
y=y0;
for i = 1:maxit
    grad = f_deriv(y,ybar,W,Wbar,u);
  
    if norm(grad)<eps
        break
    end
    y = y - alpha.*grad;
end

