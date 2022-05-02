%funzione per trovare l'alpha esatto rispetto alla loss che abbiamo

function [alpha]=exact_line_search(y,y_samp,W,W_samp,grad)
% grad è la direzione su cui stiamo cercando l'alpha ottimale
% a partire da y

u=len(y);
l=len(y_samp);
num=0;
den=0;
for i=1:u
    num = num + W_samp(:,i)'*(y(i)-y_samp)*grad(i) + W(i,:)*(y(i)-y)*grad(i);
    den = den + sum(W_samp(:,i))*grad(i)^2 + W(i,:)*(grad(i)-grad)*grad(i);
end

alpha=-num/den;

end


