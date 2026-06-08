import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kronos_system.config import PCA_VARIANCE_THRESHOLD, PCA_MAX_COMPONENTS, PCA_MIN_COMPONENTS

logger = logging.getLogger(__name__)


class CausalPCA:
    """StandardScaler + PCA with fit on training set only, per-fold.
    
    Usage:
        cpca = CausalPCA()
        train_pca = cpca.fit_transform(train_raw)   # fit + transform on train
        test_pca = cpca.transform(test_raw)          # transform only on test
        n_components = cpca.n_components_            # selected automatically
    """

    def __init__(self, variance_threshold: float = PCA_VARIANCE_THRESHOLD,
                 max_components: int = PCA_MAX_COMPONENTS,
                 min_components: int = PCA_MIN_COMPONENTS):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.min_components = min_components
        self.scaler_ = None
        self.pca_ = None
        self.n_components_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler + PCA on X, return transformed X."""
        n_samples = X.shape[0]
        max_c = min(self.max_components, n_samples - 1, X.shape[1])
        if max_c < self.min_components:
            logger.warning("Too few samples (%d) for PCA, using raw features", n_samples)
            self.scaler_ = StandardScaler()
            self.pca_ = None
            self.n_components_ = X.shape[1]
            return self.scaler_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        pca = PCA(random_state=42)
        pca.fit(X_scaled)

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n = int(np.searchsorted(cumsum, self.variance_threshold) + 1)
        n = max(self.min_components, min(n, max_c))
        self.n_components_ = n

        self.pca_ = PCA(n_components=n, random_state=42)
        result = self.pca_.fit_transform(X_scaled)
        logger.debug("PCA: %d components explains %.1f%% variance (threshold %.0f%%)",
                     n, cumsum[n - 1] * 100, self.variance_threshold * 100)
        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using fitted scaler + PCA."""
        if self.scaler_ is None:
            raise RuntimeError("CausalPCA not fitted yet. Call fit_transform first.")
        X_scaled = self.scaler_.transform(X)
        if self.pca_ is None:
            return X_scaled
        return self.pca_.transform(X_scaled)

    def get_n_components(self) -> int:
        return self.n_components_ if self.n_components_ else 0
