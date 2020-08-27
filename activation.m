function y = activation(beta, x, type)
%ACTIVATION function
if  type  == "sigmoid"
    y = (exp(-2 * beta * x) + 1) .^ (-1);
elseif type == "ReLU"
    y = (x +  abs(x))/2;
end
        
end

