% Cyclic BCGD method

function[y, tmul] = BCGD_cyclic_fixed_plot(alpha,y0,maxit,eps,ybar,W,Wbar,u)

% choose a starting point
y = y0;

for k = 1:maxit
    tic;
    % condition to stop
    grad = f_deriv(y,ybar,W,Wbar,u);

    Norm=norm(grad);
    Norms(k)=Norm;
    if Norm<eps
        break
    end

    z = y;

    for i = 1:u
       
        grad_z = f_deriv_block(z,ybar,W,Wbar,i);
        z(i) = z(i) - alpha.*grad_z;

    end

    % update
    y = z;

    
    
    Time(k)=toc;
end 
for k=1:length(Time)
    tmul(k)=sum(Time(1:k));
end
figure(1)
semilogy(Norms)
title('Norms Plot')
xlabel('iterations')
ylabel("Gradiet's norm")
figure(2)
plot(tmul)
title('Time Plot')
xlabel('iterations')
ylabel('comulative time')