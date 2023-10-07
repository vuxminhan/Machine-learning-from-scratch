import numpy as np

"""
Implementation of Linear Regression using Gradient Descent.

Let m = #training examples, n = #number of features Sizes differ 
a little bit from blog notation. It takes as input the following: 
y is R^(1 x m), X is R^(n x m), w is R^(n x 1)


"""

class LinearRegression():
    def __init__(self) -> None:
        self.learning_rate = 0.01
        self.n_iterations = 1000
        self.w = None
        
    def y_hat(self, X, w):
        return np.dot(w.T, X)
    
    def loss(self, y, y_hat):
        L = 1/self.m * np.sum((y_hat - y)**2)
        return L
    
    def gradient_descent(self, X, y, w, y_hat):
        # shape y 1xm, shape x nxm, shape w nx1
        # get nx1 matrix
        dl_dw = 2/self.m * np.dot(X, (y_hat - y).T)
        w = w - self.learning_rate * dl_dw
        return w
    
    def main(self,X,y):
        #add x1 = 1
        x1 = np.ones((1,X.shape[1]))
        X = np.append(x1, X, axis=0)
        
        self.m = X.shape[1]
        self.n = X.shape[0]
        
        w = np.zeros((self.n,1))
        
        for iter in range(self.n_iterations + 1):
            y_hat = self.y_hat(X,w)
            loss = self.loss(y_hat, y)
            
            if iter % 2000 == 0:
                print(f"Cost at iter {iter} is {loss}")
            
            w = self.gradient_descent(X, y, w, y_hat)
            
        return w

if __name__ == "__main__":
    X = np.random.rand(1, 500)
    y = 3 * X + 5 + np.random.randn(1, 500) * 0.1
    regression = LinearRegression()
    w = regression.main(X, y)
                