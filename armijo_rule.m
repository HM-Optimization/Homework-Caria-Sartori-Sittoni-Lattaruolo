% Armijo Rule

function[alpha] = armijo_rule(delta, gamma, grad, x, y_samp, W, W_samp)

D = 10;
m = 0;
fx = f(x, y_samp, W, W_samp);

while f(x + delta^m*D*(-grad), y_samp, W, W_samp) > fx - gamma*delta^m*D*(grad'*grad)
    m = m+1;
end 

alpha = delta^m*D;

end
