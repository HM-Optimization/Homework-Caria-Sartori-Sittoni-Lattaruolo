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
       
        grad_z = f_deriv_block(z,ybar,W,Wbar,i);
        z(i) = z(i) - alpha.*grad_z;

    end

    % update
    y = z;

end 
