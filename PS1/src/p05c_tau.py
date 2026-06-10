import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training and validation sets
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    # Variables to track the best model performance
    best_tau = None
    best_mse = float('inf')

    # Search tau_values for the best tau
    for tau in tau_values:
        # Re-initialize the model with the current tau in the loop
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        
        # Get predictions and calculate MSE
        preds = model.predict(x_eval)
        mse = np.mean((preds - y_eval) ** 2)
        print(f"Tau: {tau} | Validation MSE: {mse}")

        # Update best tau if this is the lowest MSE we've seen
        if mse < best_mse:
            best_mse = mse
            best_tau = tau

        # Plot predictions for this specific tau
        plt.figure()
        plt.scatter(x_train[:, 1], y_train, color='blue', marker='x', label='Training Set', s=30)
        plt.scatter(x_eval[:, 1], preds, color='red', marker='o', label='Validation Predictions', s=30)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'LWR Validation (tau = {tau}) | MSE = {mse:.4f}')
        plt.legend()
        
        # Save each plot to verify how the curve changes with tau
        plt.savefig(f'../output/plots/p05c_tau_{tau}.png', bbox_inches='tight')
        plt.close() # Close the figure so they don't pile up in memory

    print(f"\nWinner! Best tau is: {best_tau} with MSE: {best_mse}")

    # --- TEST SET PHASE ---
    # Load the test set
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # Fit a LWR model with the best tau value
    best_model = LocallyWeightedLinearRegression(best_tau)
    best_model.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    test_preds = best_model.predict(x_test)
    test_mse = np.mean((test_preds - y_test) ** 2)
    print(f"Final Test MSE: {test_mse}")

    # Save predictions to pred_path using NumPy
    np.savetxt(pred_path, test_preds)

    # Plot final test data and predictions
    plt.figure()
    plt.scatter(x_train[:, 1], y_train, color='blue', marker='x', label='Training Set', s=30)
    plt.scatter(x_test[:, 1], test_preds, color='red', marker='o', label='Test Predictions', s=30)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Final Test Set Predictions (tau = {best_tau})')
    plt.legend()
    plt.savefig('../output/plots/p05c_test_final.png', bbox_inches='tight')
    plt.show()
    # *** END CODE HERE ***