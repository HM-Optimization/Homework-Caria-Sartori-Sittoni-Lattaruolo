% Cyclic BCGD method

function[y] = BCGD_cyclic_fixed(alpha,y0,maxit,eps,ybar,W,Wbar,u)

% choose a starting point
y = y0;

for k = 1:maxit

    % condition to stop
    grad = f_deriv(y,ybar,W,Wbar,u);
    if norm(grad)<eps
        break
    end

    z = y;

    for i = 1:u
        U = zeros([u,1]);
        U(i,1) = 1;
        grad_z = f_deriv(z,ybar,W,Wbar,u);
        z = z - alpha.*U.*grad_z;

    end

    % update
    y = z;

end 