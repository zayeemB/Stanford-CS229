import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path, plot_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    model = GDA()
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    
    util.plot(x_train, y_train, model.theta, plot_path)
    np.savetxt(pred_path, pred, delimiter=",")
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        mask = y == 1
        positive_instances = x[mask]
        negative_instances = x[~mask]

        phi = len(positive_instances)/len(x)
        mu_1 = np.sum(positive_instances, axis=0)/np.sum(mask)
        mu_0 = np.sum(negative_instances, axis=0)/np.sum(~mask)

        X_centered = np.copy(x)
        X_centered[~mask] -= mu_0
        X_centered[mask] -= mu_1
        m = x.shape[0]
        sigma = (X_centered.T @ X_centered)*1/m

        sigma_inv = np.linalg.pinv(sigma)
    
        theta = sigma_inv @ (mu_1 - mu_0)
        
        theta_0 = np.log(phi / (1 - phi)) \
                - 0.5 * (mu_1.T @ sigma_inv @ mu_1) \
                + 0.5 * (mu_0.T @ sigma_inv @ mu_0)
        self.theta = np.concatenate(([theta_0], theta))
        return self.theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        probabilities = 1 / (1 + np.exp(-(x @ self.theta)))
        
        # .astype(int) turns True/False into 1/0
        predictions = (probabilities >= 0.5).astype(int) 
        
        return predictions
        # *** END CODE HERE
