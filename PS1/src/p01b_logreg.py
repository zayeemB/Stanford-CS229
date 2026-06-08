import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path, plot_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    util.plot(x_train, y_train, model.theta, plot_path)
    np.savetxt(pred_path, pred, delimiter=",")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def J_prime(theta): 
            # Below gives a (m,) matrix whose row i
            # corresponds to theta dot xi
            h_theta = x@theta
            logistic = 1/(1+np.exp(-h_theta))
            return x.T@(logistic - y)

        def J_hessian(theta):
            h_theta = x@theta
            logistic = 1/(1+np.exp(-h_theta))
            return (x.T*(logistic)*(1-logistic))@x            
     
        self.theta = np.zeros(x.shape[1])
        theta_prev = self.theta
        itr = 0
        while(itr < self.max_iter):
            self.theta = self.theta - np.linalg.inv(J_hessian(self.theta))@J_prime(self.theta)
            theta_prev = self.theta
            if np.linalg.norm(self.theta - theta_prev, ord=1) >= self.eps:
                break
            itr+=1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        probabilities = 1/(1+np.exp(-x@self.theta))
        
        return probabilities
        # *** END CODE HERE ***