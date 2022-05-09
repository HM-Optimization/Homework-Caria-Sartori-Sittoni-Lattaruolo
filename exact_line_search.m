function [alpha]=exact_line_search(y,y_samp,W,W_samp,grad)

% EXACT LINE SEARCH FUNCTION

% INPUTS
% y: starting point for the line search
% grad: direction of the line search
% y_samp: parameter of the function
% W,W_samp: weights of the function

% OUTPUT
% alpha: optimal stepsize

if length(grad)~=1
    u=length(y);
    l=length(y_samp);

    d=grad;
    num=ones(1,l)*W_samp*(y.*d) - y_samp'*W_samp*d + ones(1,u)*W*(y.*d) - y'*W*d;
    den=ones(1,l)*W_samp*(d.^2) +  ones(1,u)*W*(d.^2) - d'*W*d;
    
    alpha=num/den;

else
    alpha=1/(grad-1);
end

end
