function [grad] = f_deriv_block(y,y_samp,W,W_samp,j)

% GRADIENT COMPUTATION FOR j-th BLOCK OF f

% INPUTS
% y: point in which calculate the gradient
% y_samp: parameters of the loss fuction
% w,W_samp: weights of the loss function
% j: size 1 block index

% OUTPUT
% grad: j-th component of gradient vector of f

sum1 = W_samp(:,j)'*(y(j,1)-y_samp);
sum2 = W(j,:)*(y(j,1)-y);

grad = 2*(sum1 + sum2);

end
