function [y, timeVec, Norms, accuracy] = ...
    BCGD_random(alpha,maxit,eps,y,y_samp,W,W_samp,step_size,y_exact,delta)

% choose a starting point
u = length(y);

switch step_size
    case 1
        for k = 1:maxit
            tic;

            % condition to stop
            grad = f_deriv(y,y_samp,W,W_samp);
            
            Norm = norm(grad);
            Norms(k) = Norm;
            
            if Norm<eps
                break
            end
        
            % pick the block with uniform probability
            i = randi([1 u]);
        
            % update
            y(i) = y(i) - alpha*grad(i);
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/u);
        end
    case 2 %armijo rule
        for k = 1:maxit
            tic;

            % condition to stop
            grad = f_deriv(y,y_samp,W,W_samp);
            
            Norm = norm(grad);
            Norms(k) = Norm;
            
            if Norm<eps
                break
            end
            z = y;
            % pick the block with uniform probability
            i = randi([1 u]);
            grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
            grad=zeros([u ,1]);
            grad(i)=grad_z;
            alpha=armijo_rule(delta, grad, y, y_samp, W, W_samp);
        
            % update
            y(i) = y(i) - alpha*grad(i);
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/u);
        end
    case 3 %exact line search        
        for k = 1:maxit
            tic;

            % condition to stop
            grad = f_deriv(y,y_samp,W,W_samp);
            
            Norm = norm(grad);
            Norms(k) = Norm;
            
            if Norm<eps
                break
            end
            z = y;
            % pick the block with uniform probability
            i = randi([1 u]);
            grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
            grad=zeros([u ,1]);
            grad(i)=grad_z;
            alpha=exact_line_search(y,y_samp,W,W_samp,grad);
        
            % update
            y(i) = y(i) - alpha*grad(i);
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/u);
        end
end
       
for k=1:length(time)
    timeVec(k)=sum(time(1:k));
end
end

  
