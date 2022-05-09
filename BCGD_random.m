function [y, timeVec, Norms, accuracy] = ...
    BCGD_random(maxit,eps,y,y_samp,W,W_samp,step_size,y_exact)
    
% BLOCK COORDINATE GRADIENT DESCENT with randomized rule

% INPUTS
% y: starting point for the method
% y_samp: parameters of the loss function 
% W and W_samp: weights of the loss function
% maxit: maximum number of iteration
% eps: tollerance for the stopping rule
% y_exact: real labels for accuracy computing
% stepsize: rule for stepsize selection
% delta: parameter for the Armijo rule

% OUTPUTS
% y: final classification
% timeVec: vector of computing time per iteration
% Norms: vector of gradient norms per iteration
% accuracy: vector of accuracy per iteration


maxit=100*maxit;
u = length(y);
W_l=sum(W_samp)';
W_u=sum(W)';
C=W_l+W_u;

Norms(1)=norm(f_deriv(y,y_samp,W,W_samp)); % we compute the norm ones every 50 iteration
switch step_size
    case 1
        lambda_max = max(eig(W));
        L=2*sqrt(max(C)^2+lambda_max^2);
        alpha=1/L;

        for k = 2:maxit+1
            tic;
            
            % pick the block with uniform probability
            i = randi([1 u]);
            grad=f_deriv_block(y,y_samp,W,W_samp,i);
            
            Norm = abs(grad);
            if u*Norm<eps & Norms(floor((k-1)/100))<eps*100
                break
            end
        
            % update
            y(i) = y(i) - alpha*grad;
            
            if mod(k,100)==0
                time(k/100)=toc;
                accuracy(k/100)=1-sum(abs(sign(y)-y_exact)/u);
                Norms(k/100)=norm(f_deriv(y,y_samp,W,W_samp));
            end

        end
    case 2 % armijo rule
    delta=0.5;   % parameter for the armijo rule

        for k = 2:maxit+1
            tic;
            % pick the block with uniform probability
            i = randi([1 u]);
            grad=f_deriv_block(y,y_samp,W,W_samp,i);
            Norm = abs(grad);
            if u*Norm<eps & Norms(floor((k-1)/100))<eps*100
                break
            end
 
            alpha=armijo_rule(delta, C(i), y, y_samp, W, W_samp);
        
            % update
            y(i) = y(i) - alpha*grad;
            
            if mod(k,100)==0
                time(k/100)=toc;
                accuracy(k/100)=1-sum(abs(sign(y)-y_exact)/u);
                Norms(k/100)=norm(f_deriv(y,y_samp,W,W_samp));
            end

        end
    case 3 % exact line search        
        for k = 2:maxit+1
            tic;
            i = randi([1 u]);
            grad=f_deriv_block(y,y_samp,W,W_samp,i);
            Norm = abs(grad);
            if u*Norm<eps & Norms(k-1)<eps*100
                break
            end
            % pick the block with uniform probability
            alpha=exact_line_search(y,y_samp,W,W_samp,C(i));
        
            % update
            y(i) = y(i) - alpha*grad;
            
            if mod(k,100)==0
                time(k/100)=toc;
                accuracy(k/100)=1-sum(abs(sign(y)-y_exact)/u);
                Norms(k/100)=norm(f_deriv(y,y_samp,W,W_samp));
            end
        end
end
       
for k=1:length(time)
    timeVec(k)=sum(time(1:k));
end
end

