function [y, timeVec, Norms, accuracy] = ...
    BCGD_cyclic(y,maxit,eps,y_samp,W,W_samp,stepsize,y_exact)
   
% BLOCK COORDINATE GRADIENT DESCENT with cyclic rule

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


u = length(y);
W_l=sum(W_samp)';
W_u=sum(W)';
C=W_l+W_u-1;


switch stepsize
    case 1
        % compute the step
        lambda_max = max(eig(W));
        L=2*sqrt(max(C)^2+lambda_max^2);
        alpha=1/L;

        for k = 1:maxit
            tic;
            % condition to stop
            grad=zeros([u,1]);

            z = y;
            for i = 1:u
                grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
                z(i) = z(i) - alpha.*grad_z;
                grad(i)=grad_z;
            end
        
            % update
            y = z;
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/length(y));
            
            %stop condition
            Norm=norm(grad);
            Norms(k)=Norm;
            if Norm<eps
                break
            end

        end 
        
    case 2  % armijo
    delta=0.5;   % parameter for the armijo rule

        for k = 1:maxit
            tic;
            grad=zeros([u,1]);

            z = y;
            for i = 1:u
                grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
                grad(i)=grad_z;
                alpha=armijo_rule(delta, C(i), y, y_samp, W, W_samp);
                z(i) = z(i) - alpha.*grad_z;
            end
        
            % update
            y = z;
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/u);

            Norm=norm(grad);
            Norms(k)=Norm;
            if Norm<eps
                break
            end
        end 

    case  3 % exact line search
        for k = 1:maxit
            tic;
            grad=zeros([u,1]);
        
            z = y;        
            for i = 1:u
                grad_z = f_deriv_block(z,y_samp,W,W_samp,i);
                grad(i)=grad_z;
                alpha=exact_line_search(y,y_samp,W,W_samp,C(i));
                z(i) = z(i) - alpha.*grad_z;
            end
        
            % update
            y = z;
            time(k)=toc;
            accuracy(k)=1-sum(abs(sign(y)-y_exact)/length(y));

            Norm=norm(grad);
            Norms(k)=Norm;
            if Norm<eps
                break
            end

        end 

end
for k=1:length(time)
    timeVec(k)=sum(time(1:k));
end
end
