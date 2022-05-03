function [y, timeVec, Norms, accuracy] = GD_fixed(alpha,maxit,eps,y,y_samp,W,W_samp,step_size, y_exact)
%step_size = metodo per il calcolo dello step size che scegliamo
% Gradient descent per stepsize fisso

time=[1];
switch step_size

    case 1  %alpha fisso

        for i = 1:maxit
            tic;
            grad = f_deriv(y,y_samp,W,W_samp);
  
            Norm = norm(grad);
            Norms(i) = Norm;
            
            if Norm<eps
                break
            end
            y = y - alpha.*grad;
            time(i)=toc;
            accuracy(i)=1-sum(abs(y-y_exact)/length(y));
        end
    case 2  %armijo
        for i=1:maxit
            tic;

            grad = f_deriv(y,y_samp,W,W_samp);
  
            Norm = norm(grad);
            Norms(i) = Norm;
            
            if Norm<eps
                break
            end
            delta=0.5;
            alpha = armijo_rule(delta,grad,y,y_samp,W,W_samp);

            y = y - alpha.*grad;
            time(i)=toc;
            accuracy(i)=1-sum(abs(y-y_exact)/length(y));
        end


    case  3 %exact line search
        for i=1:maxit
            tic;
            grad = f_deriv(y,y_samp,W,W_samp);
  
            Norm = norm(grad);
            Norms(i) = Norm;
            
            if Norm<eps
                break
            end

            alpha = exact_line_search(y,y_samp,W,W_samp,grad);
           
            y = y - alpha.*grad;
            time(i)=toc;
            accuracy(i)=1-sum(abs(y-y_exact)/length(y));
        end
        
end
for k=1:length(time)
    timeVec(k)=sum(time(1:k));
end
