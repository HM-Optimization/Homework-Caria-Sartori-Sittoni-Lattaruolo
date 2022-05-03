% Cyclic BCGD method

function[y] = BCGD_cyclic_fixed(alpha,y0,maxit,eps,y_samp,W,W_samp)

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

    z = y;

    for i = 1:u
       
        grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
        z(i) = z(i) - alpha.*grad_z;

    end

    % update
    y = z;

end 
