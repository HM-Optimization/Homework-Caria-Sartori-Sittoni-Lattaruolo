% Randomized BCGD method

function[y] = BCGD_rand_fixed_uniform(alpha,y0,maxit,eps,ybar,W,Wbar,u)

% choose a starting point
y = y0;

for k = 1:maxit

    % condition to stop
    grad = f_deriv(y,ybar,W,Wbar,u);
    if norm(grad)<eps
        break
    end

    % pick the block with uniform probability
    U = zeros([u,1]);
    U(randi([1 u]),1) = 1;

    % update
    y = y - alpha.*U.*grad;

end 