% Cyclic BCGD method

function [y] = BCGD_cyclic_fixed(alpha,y0,maxit,eps,y_samp,W,W_samp)

% choose a starting point
y = y0;
u=len(y0);

for k = 1:maxit

    % condition to stop
    grad = f_deriv(y,y_samp,W,W_samp);
    if norm(grad)<eps
        break
    end

    z = y;

    for i = 1:u
        U = zeros([u,1]);
        U(i,1) = 1;  % con i blocchi di dimensione 1 questo vettore si può omettere?
        % U.*grad_z = grad_z(i) ??
        grad_z = f_deriv(z,y_samp,W,W_samp,u);
        z = z - alpha.*U.*grad_z;

    end

    % update
    y = z;

end 
