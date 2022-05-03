function[y] = BCGD_rand_fixed_uniform(alpha,y0,maxit,eps,y_samp,W,W_samp)

% choose a starting point
y = y0;
u = length(y0);

for k = 1:maxit

    % condition to stop
    grad = f_deriv(y,y_samp,W,W_samp);
    
    Norm = norm(grad);
    Norms(k) = Norm;
    
    if Norm<eps
        break
    end

    % pick the block with uniform probability
    i = randi([1 u]);

    % update
    y(i) = y(i) - alpha*grad(i);

end
