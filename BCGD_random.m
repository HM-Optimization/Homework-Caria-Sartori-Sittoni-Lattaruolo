function [y, timeVec, Norms, accuracy] = ...
    BCGD_random(maxit,eps,y,y_samp,W,W_samp,step_size,y_exact,delta)

maxit=100*maxit;
u = length(y);
for i=1:u
    W_l(i) = sum(W_samp(:,i));
    W_u(i) = sum(W(:,i));
end
C=W_l+W_u;

Norms(1)=norm(f_deriv(y,y_samp,W,W_samp)); % we compute the norm ones every 50 iteration
switch step_size
    case 1
        lambda_max = max(eig(W));
        L=sqrt(max(C)^2+lambda_max^2);
        alpha=1/L;

        for k = 2:maxit+1
            tic;
            
            % pick the block with uniform probability
            i = randi([1 u]);
            grad=f_deriv_block(y,y_samp,W,W_samp,i);
            
            Norm = abs(grad);
            if u*Norm<eps
                break
            end
        
            % update
            y(i) = y(i) - alpha*grad;
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/u);
            if mod(k,50)==0
                Norms(k)=norm(f_deriv(y,y_samp,W,W_samp));
            else
                Norms(k)=Norms(k-1);
            end

        end
    case 2 %armijo rule
        for k = 2:maxit+1
            tic;
            % pick the block with uniform probability
            i = randi([1 u]);
            grad=f_deriv_block(y,y_samp,W,W_samp,i);
            Norm = abs(grad);
            if u*Norm<eps
                break
            end
 
            alpha=armijo_rule(delta, C(i), y, y_samp, W, W_samp);
        
            % update
            y(i) = y(i) - alpha*grad;
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/u);

            if mod(k,50)==0
                Norms(k)=norm(f_deriv(y,y_samp,W,W_samp));
            else
                Norms(k)=Norms(k-1);
            end

        end
    case 3 %exact line search        
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
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/u);
            if mod(k,50)==0
                Norms(k)=norm(f_deriv(y,y_samp,W,W_samp));
            else
                Norms(k)=Norms(k-1);
            end
        end
end
       
for k=1:length(time)
    timeVec(k)=sum(time(1:k));
end
end

