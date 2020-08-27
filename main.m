% XOR function using a 2 layer perceptron. Here simple gradient descent
% method is used, therefore the convergence is very slow.
% NOTE: best combination of hyper parameters: 
% Activation_function = "sigmoid" , beta = 0.5 ,  eta = 0.3   ,  m = 0 or 1 (beta = 0.5 is the original sigmoid)
% Activation_function = "sigmoid" , beta = 7   ,  eta = 0.07  ,  m = 0 or 1 (beta = 7 is close to a sign function)
% Activation_function = "ReLU" , eta = 0.01  ,  m = 0 
% Activation_function = "ReLU" , eta = 0.03  ,  m = 1 

clear all, hold off;
% Parameters
e = [];
N0 = 2;                     % input('Number of input units = ');
N1 = 2;                     % input('Number of hidden units = ');
N2 = 1;                     % input('Number of output units = ');
eta = 0.07;                % input('Learning rate = ');
activation_function = "sigmoid" % "sigmoid", or "ReLU"
beta = 7;                   % input('beta = '); % this is beta of the sigmoid

disp('******** Initialization ********')
xi = [0 0; 0 1; 1 0; 1 1];  % input('binary pattern in Nx2 matrix form [1 0; 1 1;... ] =')
if size(xi,2) ~= N0
    error('The number of input columns should match the number of input units')
end
zeta = [0; 1; 1; 0];        % input('desired output ([1; 0; 0; 1]) = ')
m = 1;                      % input('with or wiothout momentum ( 0 or 1) = ')
alpha = 0.99;               % input('moment factor = ')
N0 = N0+1;                  % One additional term for the bias 
N1 = N1+1;                  % One additional term for the bias
[r c] = size(xi);
th = -1 * ones(r,1);
xi = [th xi];
[a b] = size(zeta);
v2 = zeros(a, b);
output = v2;
moment1 = zeros(N1, N0);
moment2 = zeros(N2, N1);
it = 0;

disp('******** MLP training starts ********')
% step 1: Initializing weights 
w1 = 0.1 * (rand(N1, N0) - 0.5);
w2 = 0.1 * (rand(N2, N1) - 0.5);

while sum(sum(abs(zeta - output))) > .1    % loop until convergence
    %it = it + 1;
    %if it > 100
    %    eta = 0.03*eta;    % learning rate annealing
    %end
    
    for i = 1:r             % Training patterns 1 through r one-by-one        
        % step 2: Forming the input to the input layer 
        v0 = xi(i,:);
        
        % step 3: compuations at the first hidden layer and the output layer 
        h1 = w1 * v0';
        v1 = activation(beta, h1, activation_function);
        v1(1) = -1;     % applying threshold
        h2 = w2 * v1;
        v2(i,:) = activation(beta, h2, activation_function);
        % MLP output, response to the ith pattern
        
        % step 4: back propagating the error to the hidden layer and finding the gradient descent steps 
        if activation_function == "sigmoid"
            gprime2 = 2 * beta * v2(i,:) .* (1- v2(i,:));
            delta2 = (gprime2) * (zeta(i,:) - v2(i,:));
        elseif activation_function == "ReLU"
            gprime2 = (sign(v2(i,:)) + 1) / 2;
            delta2 = (gprime2) * (zeta(i,:) - v2(i,:));
        end
        
        % step 5: back propagating the error to the input layer and finding the gradient descent steps 
        if activation_function == "sigmoid"
            gprime1 = 2 * beta * v1 .* (1- v1);
            delta1 = delta2 * (w2 .* (gprime1)');
        elseif activation_function == "ReLU"
            gprime1 = (sign(v1) + 1) / 2;
            delta1 = delta2 * (w2 .* (gprime1)');
        end
        
        % step 6: adjusting the weights 
        dw1 = eta * delta1' * v0;
        dw2 = eta * delta2 * v1;
        w1 = w1 + dw1;
        w2 = w2 + dw2';
        
    end
    w1 = w1 + m * alpha * moment1;
    w2 = w2 + m * alpha * moment2;
    moment1 = dw1;
    moment2 = dw2';
    
    %w2
    output = v2;
    sum(sum((zeta-output).^2))
    e = [e sum(sum((zeta-output).^2))];
    plot(e)
end
xlabel('iteration')
ylabel('error')
disp('******** MLP training is done ********')

disp('******** taking a test pattern ********')
xi1 = input ('Enter a binary pattern in Nx2 matrix form [1 0; 1 1;... ] = ');
[r c] = size(xi1);
th = -1 * ones(r,1);
v0  = [th xi1];

h1 = w1 * v0';
v1 = activation(beta, h1, activation_function);
v1(1,:) = -1;     % applying threshold
h2 = w2 * v1;
v2 = activation(beta, h2, activation_function);
disp('XOR of ') ,  xi1               
disp('is     ') ,  v2'
        
        


