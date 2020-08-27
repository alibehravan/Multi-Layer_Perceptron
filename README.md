# Multi-Layer_Perceptron
Implementation of a two layer perceptron to learn and predict simple input-output vectors in MATLAB.

The activation function can be chosen from Sigmoid or ReLU. For back-propagation a simple gradient-descent method is used, so convergence of the training is sensitive to hyper parameteres.

One of the following paramter setting can be used  for faster convergence: 
1. Activation_function = "sigmoid" , beta = 0.5 ,  eta = 0.3   ,  m = 0 or 1 (beta = 0.5 is the original sigmoid)
2. Activation_function = "sigmoid" , beta = 7   ,  eta = 0.07  ,  m = 0 or 1 (beta = 7 is close to a sign function)
3. Activation_function = "ReLU" , eta = 0.01  ,  m = 0 
4. Activation_function = "ReLU" , eta = 0.03  ,  m = 1 

