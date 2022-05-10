function [y, timeVec, Norms, accuracy] = ...
    GD(maxit,eps,y,y_samp,W,W_samp,step_size, y_exact)
%step_size = metodo per il calcolo dello step size che scegliamo
% Gradient descent 

u=length(y);
time=[1];


switch step_size

    case 1  %alpha fisso
        W_l=sum(W_samp)';
        W_u=sum(W)';
        C=W_l+W_u;
        lambda_max = max(eig(W));
        L=2*sqrt(max(C)^2+lambda_max^2);
        alpha=1/L;   % per lo step_size fisso
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
            accuracy(i)=1-sum(abs(sign(y)-y_exact)/u);
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
            alpha = armijo_rule(grad,y,y_samp,W,W_samp);

            y = y - alpha.*grad;
            time(i)=toc;
            accuracy(i)=1-sum(abs(sign(y)-y_exact)/u);
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
            accuracy(i)=1-sum(abs(sign(y)-y_exact)/u);
        end
        
end
for k=1:length(time)
    timeVec(k)=sum(time(1:k));
end

end
