function[grad] = f_deriv(y,y_samp,W,W_samp)

% GRADIENT COMPUATION FOR f 
% INPUTS
% y: point in which calcolate the gradient
% y_samp: parameters of the loss fuction
% w,W_samp: weights of the loss function
% OUTPUT
% grad: gradient vector of f

u=length(y);
l=length(y_samp);

grad=(W_samp'*ones(l,1)+W'*ones(u,1)).*y-(W_samp'*y_samp+W'*y);
grad=2*grad;

end
