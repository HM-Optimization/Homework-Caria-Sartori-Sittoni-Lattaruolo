% Armijo Rule
function [alpha] = armijo_rule(delta, grad, y, y_samp, W, W_samp)

gamma = 0.25;
D = 10;
m = 0;
fx = f(y, y_samp, W, W_samp);

while f(y + delta^m*D*(-grad), y_samp, W, W_samp) > fx - gamma*delta^m*D*(grad'*grad)
    m = m+1;
end 

alpha = delta^m*D;

end
