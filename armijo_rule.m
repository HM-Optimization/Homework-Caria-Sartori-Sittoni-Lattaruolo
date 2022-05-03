% Armijo Rule

function [alpha] = armijo_rule(delta, gamma, grad, y, y_samp, W, W_samp)

D = 10;
m = 0;
fy = f(y, y_samp, W, W_samp);

while f(y + delta^m*D*(-grad), y_samp, W, W_samp) > fy - gamma*delta^m*D*(grad'*grad)
    m = m+1;
end 

alpha = delta^m*D;

end
