import matplotlib.pyplot as plt
import numpy as np
import argparse
from src.util import load_csv

def calc_grad(X, Y, theta, lambda_=0.01):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    # Standard base gradient
    margins = (Y * X.dot(theta))
    probs = 1. / (1 + np.exp(margins))
    base_grad = -(1./m) * (X.T.dot(probs * Y))

    # L2 Regularization Penalty
    theta_penalty = np.copy(theta)
    theta_penalty[0] = 0.0  # CRITICAL: Never regularize the intercept!
    reg_grad = (lambda_ / m) * theta_penalty

    # Return the combined gradient
    return base_grad + reg_grad

def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while i < 10e5:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print(f"Loss: {calc_loss(X,Y,theta)}")
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return theta

def calc_loss(X, Y, theta):
    m, n = X.shape
    margins = (Y * X.dot(theta))
    probs = 1. / (1 + np.exp(-margins))
    logs = np.log(probs)
    return -np.sum(logs) / m


def plot_data(x, y, theta=None, save_path=None):
    neg_mask = (y == -1)

    neg_x = x[neg_mask]
    pos_x = x[~neg_mask]

    fig, ax = plt.subplots()

    ax.scatter(neg_x[:, 1], neg_x[:, 2], color="blue", marker="x", s=10, label="Negative (-1)")
    ax.scatter(pos_x[:, 1], pos_x[:, 2], color="red", marker="o", s=10, label="Positive (1)")

    ax.legend()

    if(theta is not None):
        x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
        x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1)
        ax.plot(x1, x2, c='red', linewidth=2)

    if(save_path is not None): plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process a CSV file.")

    # Add your arguments
    parser.add_argument("--path", type=str, required=True, help="The file path to the CSV document")
    parser.add_argument("--save_path", type=str, required=False, default=None)

    args = parser.parse_args()

    csv_file_path = args.path
    save_path = args.save_path

    x, y = load_csv(csv_file_path, add_intercept=True)
    theta = logistic_regression(x, y)

    plot_data(x, y, theta, save_path)

if __name__ == "__main__":
    main()