import numpy as np
import matplotlib.pyplot as plt
import pdb


class IndependentComponentAnalysis:
    """IndependentComponentAnalysis(n_components, max_iter=1000, tol=1e-5)
    ---
    Problem Statement
    Given X: n_sources x n_samples st. X = A @ S where A is the mixing matrix
    find the demixing matrix W st. S = W @ X ==> W = Inv(A)
    """

    def __init__(self, n_components, max_iter=1000, tol=1e-5):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.components_ = None

    def _center(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        return X - mean

    def _whiten(self, x):
        cov = np.cov(x)
        d, E = np.linalg.eigh(cov)
        D = np.diag(d)
        D_inv = np.sqrt(np.linalg.inv(D))
        x_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, x)))
        return x_whiten

    def _g(self, x):
        """Assumed CDF of srouce
        Can be any function that goes from 0 to 1 except for a gaussian CDF
        """
        return np.tanh(x)

    def _g_prime(self, x):
        return 1.0 - self._g(x) * self._g(x)

    def update_w(self, w, X):
        """Compute the new demixing matrix W 
        """
        w_new = (X * self._g(np.dot(w.T, X))).mean(axis=1) \
                 - self._g_prime(np.dot(w.T, X)).mean() * w
        w_new /= np.sqrt((w_new**2).sum())
        return w_new

    def _ica_g(self, X, iterations=100, tolerance=1e-5):
        if iterations > self.max_iter:
            iterations = self.max_iter
        n_sources = X.shape[0]
        W = np.zeros((n_sources, n_sources), dtype=X.dtype)
        for i in range(n_sources):
            w = np.random.rand(n_sources)
            for j in range(iterations):
                w_new = self.update_w(w, X)
                if i >= 1:
                    w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
                distance = np.abs(np.abs((w * w_new).sum()) - 1)
                w = w_new
                if distance < tolerance:
                    break
            W[i, :] = w
        # S = np.dot(W, X)
        return W

    def fit(self, X):
        X_centered = self._center(X)
        X_white = self._whiten(X_centered)
        self.components_ = self._ica_g(X_white)

    def transform(self, X):
        return np.dot(self.components_, self._whiten(self._center(X)))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def generate_data(n_samples=2000, n_sources=2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate random mixing matrix
    A = np.random.rand(n_sources, n_sources)
    max_t = 8

    # Generate sine wave source
    t = np.linspace(0, max_t, n_samples)
    sine_wave = np.sin(2 * t)  # Sine wave with frequency 1 Hz

    # Generate sawtooth wave source
    # sawtooth_wave = np.linspace(-1, 1, n_samples) % 2 - 1
    sawtooth_wave = (np.linspace(0, max_t, n_samples) * 1) % 1

    # Generate Square wave
    square_wave = np.sign(np.sin(3 * t))

    # Combine sources
    sources = np.vstack((sine_wave, sawtooth_wave, square_wave))

    # Mix sources
    mixed_sources = np.dot(A, sources)

    return mixed_sources, sources, A


def plot_signals(signals, title):
    plt.figure(figsize=(10, 6))
    for i, signal in enumerate(signals):
        plt.subplot(len(signals), 1, i + 1)
        plt.plot(signal)
        plt.title(title + ' ' + str(i + 1))
    plt.tight_layout()
    plt.show()


def plot_ica_results(mixed_sources, estimated_sources, original_sources):
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(mixed_sources.T)
    plt.title('Mixed Sources')

    plt.subplot(3, 1, 2)
    plt.plot(estimated_sources.T)
    plt.title('ICA Estimated Sources')

    plt.subplot(3, 1, 3)
    plt.plot(original_sources.T)
    plt.title('True Sources')

    plt.tight_layout()
    plt.show()


def demo_ica(n_sources=2, n_samples=2000, seed=None):
    # Generate synthetic data
    mixed_sources, true_sources, mixing_matrix = generate_data(
        n_samples, n_sources, seed)

    # Plot mixed signals
    # plot_signals(mixed_sources, title='Mixed Signals')

    # Perform ICA
    ica = IndependentComponentAnalysis(n_sources, max_iter=5000, tol=1e-8)
    estimated_sources = ica.fit_transform(mixed_sources)

    # Plot ICA results
    plot_ica_results(mixed_sources, estimated_sources, true_sources)


if __name__ == '__main__':
    demo_ica(n_sources=3, n_samples=2000, seed=42)
