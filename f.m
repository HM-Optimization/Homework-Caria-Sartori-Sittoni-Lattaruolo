% function f

function [loss] = f(y, y_samp, W, W_samp)

l = length(y_samp);
u = length(y);

loss=0;
for j = 1:u
    loss = loss + ((y_samp-y(j)).^2)'*W_samp(:,j) + 0.5*W(j,:)*(y-y(j)).^2;
end

