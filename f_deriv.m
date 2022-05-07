function[grad] = f_deriv(y,y_samp,W,W_samp)

u=length(y);
l=length(y_samp);

grad=(W_samp'*ones(l,1)+W'*ones(u,1)).*y-(W_samp'*y_samp+W'*y);
grad=2*grad;

end
