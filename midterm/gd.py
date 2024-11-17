import numpy as np

class GradientDescent:
    def __init__(self, X0, lr, max_iter, epsilon):
        self.X0 = np.array(X0)
        self.lr = lr
        self.max_iter = int(max_iter)
        self.epsilon = epsilon

    @staticmethod
    def f(X):
        return np.sum(X**2) + np.array([1, -2, -3]) @ X - np.exp(-np.sum(X**2))
    
    @staticmethod
    def nabla(X):
        return 2 * X + np.array([1, -2, -3]) + 2 * np.exp(-np.sum(X**2)) * X

    def fit(self):
        X = self.X0.copy()
        for i in range(self.max_iter):
            grad = self.nabla(X)
            X -= self.lr * grad
            if not (i+1) % (self.max_iter // 10):
                print("Iteration{}/{}\tX: {}\tf(X):{:.6f}".format(i+1, self.max_iter, X, self.f(X)))
            if np.linalg.norm(grad) <= self.epsilon:
                print("Early Stopping\nIteration{}/{}\tX: {}\tf(X):{:.6f}".format(i+1, self.max_iter, X, self.f(X)))
                break
        return X


def main():
    gd = GradientDescent(X0 = [.5, .5, .5], lr = .001, max_iter = 1e4, epsilon = 1e-4)
    gd.fit()

if __name__ == "__main__":
    main()