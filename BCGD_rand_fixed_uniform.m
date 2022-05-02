% Randomized BCGD method

function [y] = BCGD_rand_fixed_uniform(alpha,y0,maxit,eps,y_samp,W,W_samp)

% choose a starting point
u=len(y0);
y = y0;

for k = 1:maxit

    % condition to stop
    grad = f_deriv(y,y_samp,W,W_samp);
    if norm(grad)<eps
        break
    end

    % pick the block with uniform probability
    U = zeros([u,1]);
    U(randi([1 u]),1) = 1;

    % update
    y = y - alpha.*U.*grad;
    
    %possiamo scrivere tutto in una sola riga di codice sfruttando
    % U.*grad = grad(randi([1 u]))  ? 

end 

end
