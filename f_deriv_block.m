function [grad] = f_deriv_block(y,y_samp,W,W_samp,j)
% j = indice del blocco 1-dim

sum1 = 0;
sum2 = 0;
sum1 = sum( W_samp(:,j).*(y(j,1)-y_samp) );
sum2 = sum( W(:,j).*(y(j,1)-y) );

grad = sum1 + sum2;

end
