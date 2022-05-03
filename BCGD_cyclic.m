% Cyclic BCGD method

function [y, timeVec, Norms, accuracy] = ...
    BCGD_cyclic(alpha,y,maxit,eps,y_samp,W,W_samp,stepsize,y_exact,delta)

u = length(y);



switch stepsize
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
        
            z = y;
        
            for i = 1:u
               
                grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
                z(i) = z(i) - alpha.*grad_z;
        
            end
        
            % update
            y = z;
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/length(y));
        end 
        
    case 2  %armijo
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
        
            for i = 1:u
                grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
                grad=zeros([u ,1]);
                grad(i)=grad_z;
                alpha=armijo_rule(delta, grad, y, y_samp, W, W_samp);
                z(i) = z(i) - alpha.*grad_z;
        
            end
        
            % update
            y = z;
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/length(y));
        end 
    case  3 %exact line search
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
        
            for i = 1:u
                grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
                grad=zeros([u ,1]);
                grad(i)=grad_z;
                alpha=exact_line_search(y,y_samp,W,W_samp,grad);
                z(i) = z(i) - alpha.*grad_z;
        
            end
        
            % update
            y = z;
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/length(y));
        end 
end
for k=1:length(time)
    timeVec(k)=sum(time(1:k));
end
end

