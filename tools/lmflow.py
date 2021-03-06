# LinearModelFlow: LMflow

import abc
from typing import Optional

import gpflow
from gpflow.utilities.ops import square_distance
import numpy as np
import tensorflow as tf


class BasisFunctionKernel(gpflow.kernels.Kernel):
    """
    If anyone ever sees this... This is not a good way to implement basis func regression..!
    Use Woodbury instead!
    """

    def __init__(self, variance=1.0, **kwargs):
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive())

    @abc.abstractmethod
    def Phi(self, X):
        raise NotImplementedError

    def K(self, X, X2=None):
        Phi = self.Phi(X)
        if X2 is None:
            return tf.matmul(Phi, Phi, transpose_b=True)
        else:
            return tf.matmul(Phi, self.Phi(X2), transpose_b=True)

    def K_diag(self, X):
        return tf.reduce_sum(self.Phi(X) ** 2.0, axis=1)


class InputScaledBasisFunctionKernel(BasisFunctionKernel):
    def __init__(self, variance=1.0, lengthscales=1.0, **kwargs):
        super().__init__(variance, **kwargs)
        self.lengthscales = gpflow.Parameter(lengthscales, transform=gpflow.utilities.positive())


class PolynomialBasisKernel(InputScaledBasisFunctionKernel):
    def __init__(self, degree, variance=1.0, lengthscales=1.0, **kwargs):
        super().__init__(variance, lengthscales, **kwargs)
        self.degree = degree

    def Phi(self, X):
        X = X / self.lengthscales
        return tf.concat([self.variance ** 0.5 * X ** tf.cast(i, X.dtype) for i in range(0, self.degree + 1)], axis=1)


class SqExpBasisFunctionKernel(InputScaledBasisFunctionKernel):
    def __init__(self, number_of_bases=10, range=(-1, 1), variance=1.0, lengthscales=1.0, **kwargs):
        super().__init__(variance, lengthscales, **kwargs)
        self.centres = tf.constant(np.linspace(range[0], range[1], number_of_bases)[:, None])

    def Phi(self, X):
        sqdist = square_distance(X, self.centres) / self.lengthscales ** 2.0
        return self.variance ** 0.5 * tf.exp(-sqdist)


class SinusoidalBasisKernel(InputScaledBasisFunctionKernel):
    def __init__(self, number_of_bases=10, variance=1.0, lengthscales=1.0, **kwargs):
        super().__init__(variance, lengthscales, **kwargs)
        self.frequencies = gpflow.Parameter(np.random.randn(number_of_bases))
        self.phases = gpflow.Parameter(np.random.rand(number_of_bases) * np.pi * 2)

    def Phi(self, X):
        X = X / self.lengthscales
        return self.variance ** 0.5 * tf.math.sin(X * self.frequencies + self.phases)
        # return tf.math.sin(X)


class LMR(gpflow.models.GPR):
    r"""
    Linear Model Regression.
    """

    def __init__(
            self,
            data: gpflow.models.training_mixins.RegressionData,
            kernel: BasisFunctionKernel,
            mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
            noise_variance: float = 1.0,
    ):
        super().__init__(data, kernel, mean_function, noise_variance)

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        err = Y - self.mean_function(X)
        N = X.shape[0]

        ?? = self.kernel.Phi(X)
        D = ??.shape[1]

        ??T?? = tf.matmul(??, ??, transpose_a=True)
        ??inv_?? = tf.eye(??.shape[1], dtype=??.dtype) / self.kernel.variance + ??T?? / self.likelihood.variance
        chol??inv_?? = tf.linalg.cholesky(??inv_??)

        const = -0.5 * N * np.log(2 * np.pi)
        logdet = -0.5 * (
                2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol??inv_??))) +
                D * tf.math.log(self.kernel.variance) +
                N * tf.math.log(self.likelihood.variance)
        )
        ??Tyvarinv = tf.matmul(??, err, transpose_a=True) / self.likelihood.variance
        ??_?? = tf.linalg.solve(??inv_??, ??Tyvarinv)
        quadratic = -0.5 * (
                tf.reduce_sum(err ** 2.0) / self.likelihood.variance +
                - tf.reduce_sum(??_?? * ??Tyvarinv)
        )

        return const + logdet + quadratic

    def predict_f(
            self, Xnew: gpflow.models.training_mixins.InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> gpflow.models.model.MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        ??_pred = self.kernel.Phi(Xnew)
        ?? = self.kernel.Phi(X_data)

        ??T?? = tf.matmul(??, ??, transpose_a=True)
        ??inv_?? = tf.eye(??.shape[1], dtype=??.dtype) / self.kernel.variance + ??T?? / self.likelihood.variance
        # It's more numerically stable to do the inverse with the Cholesky and then a triangular solve.
        ??_?? = tf.linalg.solve(??inv_??, tf.matmul(??, err, transpose_a=True)) / self.likelihood.variance

        f_mean_zero = ??_pred @ ??_??
        ??inv_??_??_pred = tf.linalg.solve(??inv_??, tf.transpose(??_pred))
        if full_cov:
            f_var = (??_pred @ ??inv_??_??_pred)[None, :, :]
        else:
            f_var = tf.reduce_sum(??_pred * tf.transpose(??inv_??_??_pred), 1)[:, None]

        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
