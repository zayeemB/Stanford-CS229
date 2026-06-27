import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    m = x.shape[0]

    rng = np.random.default_rng()

    k = K
    # Randomly assigns a label to xi
    arr = rng.integers(low=1, high=k, size=m, endpoint=True)
    
    label = np.arange(1, k+1)
    # (m,k) array whose j colum has 1's corresponding
    # to xi that have been assigned label j
    mask = label[np.newaxis, :] == arr[:, np.newaxis]

    mu = mask.T @ x/ (np.sum(mask, axis=0))[:, np.newaxis]

    # (k,m,n), jth matrix is the examples corresponding to jth group
    # Only xi (rows) corresponding to group j are non zero
    # All others are zero vectors
    diff = (x[np.newaxis, :, :] - mu[:, np.newaxis, :]) * mask.T[:, :, np.newaxis]

    # sum_i->Nj((x_i - mu_j)@(x_i - mu_j).T)
    # (k,n,n) resulting matrix
    inner_prods_sum = np.einsum("ijk, ijl -> ikl", diff, diff)

    sigma = inner_prods_sum / (np.sum(mask, axis=0))[:, np.newaxis, np.newaxis]

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(k)
    phi = phi / k

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((m,k))
    w = w/k

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        #------(1) E-step: Update your estimates in w------

        sigma_inv = np.linalg.inv(sigma)

        mu = np.array(mu)

        # The i,j row of the (m,n,n) array is the
        # difference of the ith row of x and jth row of mu
        diff = x[:, np.newaxis, :] - mu[np.newaxis, :, :]

        # diff,ij.T @ sigma_inv,j @ diff,ij
        # (m,k) matrix
        prod = np.einsum('mki, kij, mkj -> mk', diff, sigma_inv, diff)
        exp = np.exp(-1/2*prod)
        # (k,) array of determinants of sigma
        n = x.shape[1]
        det = 1 / np.sqrt(np.linalg.det(sigma) * (2*np.pi)**n)

        un_w = det * exp * phi
        # Finally w is a (m,k) matrix
        w = un_w/(un_w.sum(axis=1))[:, np.newaxis]

        #------(2) M-step: Update the model parameters phi, mu, and sigma------

        m = x.shape[0]

        # Update for phi
        phi = np.sum(w, axis=0).T/m
        # Update for mu
        mu = (w.T @ x)/np.sum(w, axis=0)[:, np.newaxis]

        # We use the updated value of mu
        diff = x[:, np.newaxis, :] - mu[np.newaxis, :, :]

        # Update for sigma
        outer_prods = np.einsum("ijk, ijp -> ijkp", diff, diff)
        summation = np.einsum("ijkp, ij -> jkp", outer_prods, w)
        sigma = summation/np.sum(w, axis=0)[:, np.newaxis, np.newaxis]

        #------(3) Compute the log-likelihood of the data to check for convergence------

        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        prev_ll = ll

        # x and mu have been updated
        sigma_inv = np.linalg.inv(sigma)

        diff = x[:, np.newaxis, :] - mu[np.newaxis, :, :]
        prod = np.einsum('mki, kij, mkj -> mk', diff, sigma_inv, diff)
        exp = np.exp(-1/2*prod)

        # sigma has been updated
        det = 1 / np.sqrt(np.linalg.det(sigma) * (2 * np.pi)**n)
        
        # (m,k) matrix
        # ij is the probability that xi comes from
        # distibution j
        probs = det * exp * phi
        log_probs = np.log(probs)

        ll = np.sum(np.log(np.sum(probs, axis=1)))

        it+=1
        # *** END CODE HERE ***
    print(f"Converged in {it} iterations.")

    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w

        sigma_inv = np.linalg.inv(sigma)

        mu = np.array(mu)

        # The i,j row of the (m,n,n) array is the
        # difference of the ith row of x and jth row of mu
        diff = x[:, np.newaxis, :] - mu[np.newaxis, :, :]

        # diff,ij.T @ sigma_inv,j @ diff,ij
        # (m,k) matrix
        prod = np.einsum('mki, kij, mkj -> mk', diff, sigma_inv, diff)
        exp = np.exp(-1/2*prod)
        # (k,) array of determinants of sigma
        n = x.shape[1]
        det = 1 / np.sqrt(np.linalg.det(sigma) * (2*np.pi)**n)

        un_w = det * exp * phi
        # Finally w is a (m,k) matrix
        w = un_w/(un_w.sum(axis=1))[:, np.newaxis]

        # (2) M-step: Update the model parameters phi, mu, and sigma

        m = x.shape[0]
        k = len(mu)
        m_tilde = x_tilde.shape[0]

        # (m_tilde, k) matrix
        # ith row has maks[i][j] = 1 if xi has z = j
        # Each colum j represents all xi that have z = j
        mask = (z == np.arange(1, k+1)).astype(int)

        # phi update
        z_sum = np.sum(mask, axis=0)
        w_sum = np.sum(w, axis=0)

        phi = (w_sum + alpha * z_sum) / (m + alpha * m_tilde)

        # mu update
        k = mu.shape[0]

        # sum_i(wij xi)
        w_x_sum = (w.T @ x)

        # alpha * sum_i(x_tilde_i 1{z_i = l})
        x_tilde_sum = alpha * (mask.T @ x_tilde)

        total_sum = alpha * z_sum + w_sum

        mu = (w_x_sum + x_tilde_sum) / total_sum[:, np.newaxis]

        # We use the updated value of mu
        diff_unlabelled = x[:, np.newaxis, :] - mu[np.newaxis, :, :]
        diff_labelled = x_tilde[:, np.newaxis, :] - mu[np.newaxis, :, :]

        # Update for sigma
        outer_prods_unlabelled = np.einsum("ijk, ijp -> ijkp", diff_unlabelled, diff_unlabelled)
        outer_prods_labelled = np.einsum("ijk, ijp -> ijkp", diff_labelled, diff_labelled)

        summation_w = np.einsum("ijkp, ij -> jkp", outer_prods_unlabelled, w)
        summation_z = np.einsum("ijkp, ij -> jkp", outer_prods_labelled, alpha * mask)

        sigma = (summation_w + summation_z) / total_sum[:, np.newaxis, np.newaxis]

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        prev_ll = ll

        # Pre-compute inverted sigma and determinant for both datasets
        sigma_inv = np.linalg.inv(sigma)
        det = 1 / np.sqrt(np.linalg.det(sigma) * (2 * np.pi)**n)
        
        # We use a microscopic epsilon to prevent np.log(0)
        eps_pad = 1e-16
        
        # A. Unlabeled Log-Likelihood
        diff = x[:, np.newaxis, :] - mu[np.newaxis, :, :]
        prod = np.einsum('mki, kij, mkj -> mk', diff, sigma_inv, diff)
        probs = det * np.exp(-0.5 * prod) * phi
        # ADD EPS_PAD HERE
        ll_unlabeled = np.sum(np.log(np.sum(probs, axis=1) + eps_pad))

        # B. Labeled Log-Likelihood
        diff_tilde = x_tilde[:, np.newaxis, :] - mu[np.newaxis, :, :]
        prod_tilde = np.einsum('mki, kij, mkj -> mk', diff_tilde, sigma_inv, diff_tilde)
        probs_tilde = det * np.exp(-0.5 * prod_tilde) * phi
        
        true_class_probs = np.sum(probs_tilde * mask, axis=1)
        # ADD EPS_PAD HERE
        ll_labeled = alpha * np.sum(np.log(true_class_probs + eps_pad))

        # Total Objective
        ll = ll_unlabeled + ll_labeled

        it+=1
        # *** END CODE HERE ***
    
    print(f"Converged in {it} iterations.")
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
