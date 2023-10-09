import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression():
    def __init__(self, learning_rate = 0.0001, n_iterations = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.losses = []
        self.delta = 0.0001
        self.w = None
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
    
    def y_hat(self, X, w):
        z =  np.dot(X, w)
        return self.sigmoid(z)
    
    def loss(self, y, y_hat):
        L = -1/self.n * np.sum(y * np.log(y_hat) + (1-y)*np.log(1-y_hat))
        return L
    
    def gradient_descent(self, X, y, w, y_hat):
        # shape y 1xm, shape x nxm, shape w nx1
        # get nx1 matrix
        dl_dw = 1/self.n * np.dot(X.T, (y_hat - y))
        w = w - self.learning_rate * dl_dw
        return w
    
    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0],1)), X]  
        self.w = np.zeros(X.shape[1])
        self.n, self.d = X.shape
        print(X.shape, y.shape, self.w.shape)
        for iter in range(self.n_iterations + 1):
            y_hat = self.y_hat(X, self.w)
            old_w = self.w.copy()
            self.w = self.gradient_descent(X, y, self.w, y_hat)
            if iter % 10000 == 0:
                loss = self.loss(y, y_hat)
                print(f'loss: {loss}')
                self.losses.append(loss)
            
            #if new w - old w < delta, stop
            if np.linalg.norm(self.w - old_w) < self.delta:
                break
        return self.w
    
    def predict(self, X, threshold = 0.5):
        X = np.c_[np.ones((X.shape[0],1)), X]
        y_hat = self.y_hat(X, self.w)
        y_hat_label = (y_hat >= threshold).astype(int)
        return y_hat_label
    
    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs. Iteration')
        plt.grid(True)
        plt.show()
        
    def mean_error(self, y_true, y_pred):
        return np.mean(y_pred == y_true)

    
    
if __name__ == "__main__":
    df_train = pd.read_csv('../../data/ds1_train.csv')
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    
    df_test = pd.read_csv('../../data/ds1_valid.csv')
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    log_reg = LogisticRegression(learning_rate=0.01, n_iterations=1000000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print(f'Accuracy: {log_reg.mean_error(y_test, y_pred)}')
    log_reg.plot_loss()
    