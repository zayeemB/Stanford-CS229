import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    _, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    logistic_model = LogisticRegression()

    logistic_model.fit(x_train, t_train)
    true_label_preds = logistic_model.predict(x_test)

    np.savetxt(pred_path_c , (true_label_preds>=0.5).astype(int), delimiter=", ")

    true_label_pred_theta = logistic_model.theta
    util.plot(x_test, t_test, true_label_pred_theta, '../output/p02c.png')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    logistic_model.fit(x_train, y_train)
    y_label_preds = logistic_model.predict(x_test)

    np.savetxt(pred_path_d, (y_label_preds>=0.5).astype(int), delimiter=', ')

    util.plot(x_test, t_test, logistic_model.theta, '../output/p02d.png')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    
    V_plus_mask = (y_val==1)
    x_val_plus = x_val[V_plus_mask]

    alpha = np.mean(logistic_model.predict(x_val_plus))
    preds_e = y_label_preds / alpha

    np.savetxt(pred_path_e, preds_e, delimiter=",")
    
    util.plot(x_test, t_test, logistic_model.theta, '../output/p02e.png', correction=alpha)
    # *** END CODER HERE
