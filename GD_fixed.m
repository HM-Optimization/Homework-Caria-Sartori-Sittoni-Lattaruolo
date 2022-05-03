function[y, timeVec, Norms, accuracy] = GD_fixed(alpha,y0,maxit,eps,y_samp,W,W_samp,u)
% Gradient descent per stepsize fisso
y=y0;
for i = 1:maxit
    grad = f_deriv(y,y_samp,W,W_samp,u);
  
    Norm = norm(grad);
    Norms(k) = Norm;
    
    if Norm<eps
        break
    end
    y = y - alpha.*grad;
end

