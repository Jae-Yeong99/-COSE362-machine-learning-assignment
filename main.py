import sys
import random
import numpy as np
import matplotlib.pyplot as plt

def load_xy(path, delimiter=None):
    try:
        data = np.loadtxt(path, delimiter=delimiter)
    except Exception:
        try:
            data = np.loadtxt(path, delimiter=',')
        except Exception:
            data = np.loadtxt(path)
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y

def std_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def std_apply(X, mean, std):
    return (X - mean) / std

def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(s, axis=axis)

def log_gaussian_pdf(X, mu, Sigma):
    D = X.shape[1]
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(D))
    diff = (X - mu)
    z = np.linalg.solve(L, diff.T)
    maha = np.sum(z**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2.0 * np.pi) + log_det + maha)

class GMM:
    def __init__(self, n_components, reg_covar=1e-6, max_iter=200, tol=1e-4, seed=42):
        self.K = n_components
        self.reg = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _init_params(self, X):
        rng = random.Random(self.seed)
        N, D = X.shape
        idx = rng.sample(range(N), self.K)
        self.mu_ = X[idx].copy()
        cov = np.cov(X, rowvar=False) + self.reg * np.eye(D)
        self.Sigma_ = np.stack([cov.copy() for _ in range(self.K)], axis=0)
        self.pi_ = np.ones(self.K) / self.K

    def _e_step(self, X):
        N = X.shape[0]
        log_resp = np.zeros((N, self.K))
        for k in range(self.K):
            log_resp[:, k] = np.log(self.pi_[k] + 1e-16) + log_gaussian_pdf(X, self.mu_[k], self.Sigma_[k])
        log_norm = logsumexp(log_resp, axis=1)
        ll = np.sum(log_norm)
        resp = np.exp(log_resp - log_norm[:, None])
        return resp, ll

    def _m_step(self, X, resp):
        N, D = X.shape
        Nk = resp.sum(axis=0) + 1e-16
        self.pi_ = Nk / N
        self.mu_ = (resp.T @ X) / Nk[:, None]
        for k in range(self.K):
            diff = X - self.mu_[k]
            self.Sigma_[k] = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
            self.Sigma_[k].flat[:: D + 1] += self.reg

    def fit(self, X):
        self._init_params(X)
        prev_ll = -1e300
        for _ in range(self.max_iter):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def score_logpdf(self, X):
        log_comp = []
        for k in range(self.K):
            log_comp.append(np.log(self.pi_[k] + 1e-16) + log_gaussian_pdf(X, self.mu_[k], self.Sigma_[k]))
        log_comp = np.stack(log_comp, axis=1)
        return logsumexp(log_comp, axis=1)

class GMMBinaryClassifier:
    def __init__(self, n_components, reg_covar=1e-6, max_iter=200, tol=1e-4, seed=42):
        self.K = n_components
        self.reg = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X, y):
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.pr0 = len(X0) / len(X)
        self.pr1 = 1.0 - self.pr0
        self.g0 = GMM(self.K, self.reg, self.max_iter, self.tol, self.seed).fit(X0)
        self.g1 = GMM(self.K, self.reg, self.max_iter, self.tol, self.seed + 1).fit(X1)
        return self

    def predict(self, X):
        log0 = self.g0.score_logpdf(X) + np.log(self.pr0 + 1e-16)
        log1 = self.g1.score_logpdf(X) + np.log(self.pr1 + 1e-16)
        return (log1 > log0).astype(int)

def kfold_indices(n_samples, n_splits, seed=42, shuffle=True):
    indices = list(range(n_samples))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    folds = np.array_split(np.array(indices), n_splits)
    splits = []
    all_idx = np.arange(n_samples)
    for k in range(n_splits):
        val_idx = folds[k]
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        splits.append((train_idx, val_idx))
    return splits

def cv_search_components_early_stop(X, y, n_splits, seed=42, reg=1e-6, max_iter=200, tol=1e-4, patience=3):
    splits = kfold_indices(len(y), n_splits, seed=seed, shuffle=True)
    val_errors = {}
    train_errors = {}

    best_val = float('inf')
    best_K = 1
    K = 1
    worsening_count = 0

    while True:
        fold_val_errs = []
        fold_train_errs = []
        for tr, va in splits:
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]
            mean, std = std_fit(Xtr)
            Xtr_s = std_apply(Xtr, mean, std)
            Xva_s = std_apply(Xva, mean, std)
            clf = GMMBinaryClassifier(K, reg_covar=reg, max_iter=max_iter, tol=tol, seed=seed)
            clf.fit(Xtr_s, ytr)
            yhat_tr = clf.predict(Xtr_s)
            train_err = 1.0 - np.mean(yhat_tr == ytr)
            fold_train_errs.append(train_err)
            yhat_va = clf.predict(Xva_s)
            val_err = 1.0 - np.mean(yhat_va == yva)
            fold_val_errs.append(val_err)

        train_errors[K] = float(np.mean(fold_train_errs))
        val_errors[K] = float(np.mean(fold_val_errs))
        print(f"K={K:2d} | Train error = {train_errors[K]:.4f} | Val error = {val_errors[K]:.4f}")

        if val_errors[K] < best_val:
            best_val = val_errors[K]
            best_K = K
            worsening_count = 0
        else:
            worsening_count += 1
            if worsening_count >= patience:
                print(f"Validation error increased {patience} times consecutively. Stopping search.")
                break

        K += 1

    return train_errors, val_errors, best_K

def plot_errors(train_errors, val_errors, out_path='cv_error_vs_components.png'):
    ks = sorted(train_errors.keys())
    train_vals = [train_errors[k] for k in ks]
    val_vals = [val_errors[k] for k in ks]
    plt.figure()
    plt.plot(ks, train_vals, marker='o', label='Train error')
    plt.plot(ks, val_vals, marker='s', label='Validation error')
    plt.xlabel('Number of Gaussian components per class (K)')
    plt.ylabel('Error (1 - accuracy)')
    plt.title('GMM model selection via K-fold CV')
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.show()

def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else 'train.txt'
    test_path  = sys.argv[2] if len(sys.argv) > 2 else 'test.txt'
    kfold      = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    X, y = load_xy(train_path)
    train_errors, val_errors, best_K = cv_search_components_early_stop(X, y, n_splits=kfold, patience=3)
    plot_errors(train_errors, val_errors)
    print(f'Best K = {best_K} (Validation error = {val_errors[best_K]:.4f})')

    mean, std = std_fit(X)
    Xs = std_apply(X, mean, std)
    clf = GMMBinaryClassifier(best_K)
    clf.fit(Xs, y)

    try:
        Xt, yt = load_xy(test_path)
        Xt = std_apply(Xt, mean, std)
        yhat = clf.predict(Xt)
        test_err = 1.0 - np.mean(yhat == yt)
        print(f'Test error = {test_err:.4f}')
    except Exception as e:
        print('Could not evaluate on test set:', e)

if __name__ == '__main__':
    main()
