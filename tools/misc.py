import gpflow
import numpy as np
from .lmflow import BasisFunctionKernel


def sample_prior(model_or_kernel, pX, num_samples=10):
    k = model_or_kernel if isinstance(model_or_kernel, gpflow.kernels.Kernel) else model_or_kernel.kernel
    if not isinstance(k, BasisFunctionKernel):
        K = k(pX)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
        samples = L @ np.random.randn(len(L), num_samples)
    else:
        Phi = k.Phi(pX)
        samples = Phi @ np.random.randn(Phi.shape[1], num_samples)
    return samples
