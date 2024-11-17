import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RFR:
    def __init__(self, data, params):
        self.dt = data
        self.params = params
        self.split(size = params["test_size"])
    
    def split(self, size):
        X_train, X_valid, self.y_train, self.y_valid = train_test_split(self.dt["x1"].to_numpy(), self.dt["x2"].to_numpy(), 
                                                                                test_size = size, 
                                                                                random_state = self.params["random_state"])
        self.X_train = X_train.reshape(-1, 1)
        self.X_valid = X_valid.reshape(-1, 1)
    
    def fit(self):
        self.model = RandomForestRegressor(n_estimators = self.params["n_est"], criterion = self.params["rf_criterion"],
                                    max_depth = self.params["max_depth"], random_state = self.params["random_state"])
        self.model.fit(self.X_train, self.y_train)
    
    @staticmethod
    def get_metrics(y, y_pred, name: str):
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mse)
        print("Set: %s\nMSE: %.4f\tMAE: %.4f\tR2 Score: %.4f\tRMSE: %.4f" %(name, mse, mae, r2, rmse))

    def evaluate(self, plot = False):
        y_train_pred = self.model.predict(self.X_train)
        y_valid_pred = self.model.predict(self.X_valid)
        self.get_metrics(self.y_train, y_train_pred, name = "Training")
        self.get_metrics(self.y_valid, y_valid_pred, name = "Validation")
        if plot:
            self.plot_predictions(self.X_train, y_train_pred)
            self.plot_predictions(self.X_valid, y_valid_pred)

    def test_rnd(self, num_points, ext_ratio = 0, plot = False):
        X_test = np.random.uniform(self.dt["x1"].min()*(1 - ext_ratio), self.dt["x1"].max()*(1 + ext_ratio), num_points).reshape(-1, 1)
        pred = self.model.predict(X_test)
        print("Generated data points: {}\nPredictions: {}".format(X_test.flatten(), pred))
        if plot:
            self.plot_predictions(X_test, pred)

    def plot_predictions(self, X, y):
        dt_sorted = self.dt.sort_values(by = "x1", axis = 0, ascending = True)
        plt.plot(dt_sorted["x1"], dt_sorted["x2"])
        plt.scatter(X, y, c = "r")
        plt.show()


def main():
    data = pd.read_csv("./midterm/predictive.csv")
    rf_model = RFR(data, params = {"test_size": .3, "random_state": 42, "n_est": 100, "max_depth": 5, "rf_criterion": "squared_error"})
    rf_model.fit()
    rf_model.evaluate()
    rf_model.test_rnd(num_points = 10, ext_ratio = .1, plot = True)

if __name__ == "__main__":
    main()