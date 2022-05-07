function [loss] = f(y, y_samp, W, W_samp)

% LOSS FUNCTION
% INPUTS 
% y, y_samp, W, W_samp: arguments of the loss fuction
% OUTPUT 
% loss: value of the function we want to minimize

l = length(y_samp);
u = length(y);

loss=ones(1,l)*W_samp*(y.^2) - 2*(y_samp'*W_samp*y) + (y_samp.^2)'*W_samp*ones(u,1) + ones(1,u)*W*(y.^2) - y'*W*y;

end
