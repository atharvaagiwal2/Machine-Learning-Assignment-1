
class Perceptron:
    
    def __init__(self, learning_rate = 0.001, epochs = 1000):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, z):
        return np.heaviside(z, 0)
    
#                       0   if x1 < 0
# heaviside(x1, x2) =  x2   if x1 == 0
# here x2 = 0
#                       1   if x1 > 0

    def fit(self, X, y):
        columns = X.shape[1]
        # This will give the number of columns is 2-d array 
        # Initializing weights and bias with zeros
        self.weights = np.zeros((columns))
        self.bias = 0
        
        # Iterating until the number of epochs
        for epoch in range(self.epochs):
            
            # Traversing through the entire training set
            for i in range(len(X)):
                z = np.dot(X, self.weights) + self.bias # Finding the dot product and adding the bias
                y_pred = self.activation(z) # Passing through an activation function
                
                # old_weight = self.weights 
                #Updating weights and bias
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred[i]) * X[i]
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])
                
                # if((old_weight==self.weights).all()):
                  # print("Weights are unchanged from the previous iteration")
                # else:
                  # print(self.weights)
        return self.weights, self.bias
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
