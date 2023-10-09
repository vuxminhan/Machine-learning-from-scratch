import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
class NaiveBayes():
    def __init__(self) -> None:
        pass
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.priors = np.zeros(self.n_classes)
        self.means = np.zeros((self.n_classes, self.d))
        self.variances = np.zeros((self.n_classes, self.d))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = X_c.shape[0]/self.n
            self.means[i,:] = X_c.mean(axis=0)
            self.variances[i,:] = X_c.var(axis=0)
        return self
    
    def predict(self, X):
        y_hat = [self._predict(x) for x in X]
        return y_hat
    
    @staticmethod
    def gaussian_pdf(X, mean, variance):
        return 1/np.sqrt(2*np.pi*variance) * np.exp(-(X - mean)**2/(2*variance))
    
    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            posterior = np.sum(np.log(self.gaussian_pdf(x, self.means[i,:], self.variances[i,:])))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
if __name__ == "__main__":

    df = pd.read_csv('../../data/ung thư.csv')
    
    # Hist sub plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flatten()):
        sns.histplot(data=df, x=df.columns[i], hue='diagnosis', kde=True, ax=ax)
        plt.title(df.columns[i])
    plt.show()
    
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.score(y_pred, y_test))

    
    