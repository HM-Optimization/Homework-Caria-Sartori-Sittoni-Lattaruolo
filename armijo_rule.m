% Armijo Rule
function [alpha] = armijo_rule(delta, grad, y, y_samp, W, W_samp)
gamma = 0.25;

if length(grad)~=1
    
    D = 1;
    m = 0;
    fx = f(y, y_samp, W, W_samp);
    
    while f(y + delta^m*D*(-grad), y_samp, W, W_samp) > fx - gamma*delta^m*D*(grad'*grad) & m<20
        m = m+1;
    end 
    
    alpha = delta^m*D;
else 
    alpha = (1-gamma)/grad;
end

end
