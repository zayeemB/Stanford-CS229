import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)

    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    preds = model.predict(x_val)
    
    mse = np.mean((preds - y_val) ** 2)
    print(f"Validation MSE: {mse}")

    plt.figure()
    
    # Plot training set with blue 'x' markers
    # We use x_train[:, 1] to grab the actual feature column and ignore the intercept column of 1s
    plt.scatter(x_train[:, 1], y_train, color='blue', marker='x', label='Training Set')
    
    # Plot validation predictions with red 'o' markers
    plt.scatter(x_val[:, 1], preds, color='red', marker='o', label='Validation Predictions')
    
    # plt.scatter(x_val[:, 1], y_val, color='gray', marker='.', alpha=0.5, label='True Validation Data')

    # Style the graph
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Locally Weighted Linear Regression ($\\tau$ = {tau})')
    plt.legend()
    
    # Render the graph onto the screen
    plt.show()
    plt.savefig("../ouput/p05b.png")
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        r_xt = self.x[np.newaxis, :, :]
        r_xe = x[:, np.newaxis, :]

        distances_sq = np.sum((r_xt - r_xe) ** 2, axis=2)
        w = np.exp(-distances_sq / (2 * self.tau ** 2))

        w_daigs = w[:, :, np.newaxis] * np.eye(w.shape[1])
        
        s = self.x.T[np.newaxis, :, :] @ w_daigs @ self.x
        r = self.x.T[np.newaxis, :, :] @ w_daigs @ self.y

        theta = np.linalg.inv(s) @ r[:, :, np.newaxis]
        theta = np.squeeze(theta, axis=-1)
        
        return np.sum(theta * x, axis=1)
        # *** END CODE HERE ***
