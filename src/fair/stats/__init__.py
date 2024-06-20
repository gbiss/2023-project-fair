import random

import numpy as np
from scipy import stats
from statsmodels.distributions.copula.api import CopulaDistribution, GaussianCopula


def bernoulli_samples(theta: np.ndarray, n: int = 1):
    """Generate Bernoulli samples from parameter vector theta

    Args:
        theta (np.ndarray): Bernoulli parameters
        n (int, optional): Number of samples. Defaults to 1.

    Returns:
        np.ndarray: nXm matrix of samples, one per row
    """
    theta = theta.flatten()
    m = len(theta)

    return np.hstack(
        [stats.bernoulli(theta[i]).rvs(n).reshape((n, 1)) for i in range(m)]
    )


def binary(integer: int, n: int) -> np.ndarray:
    """Convert integer to binary

    Args:
        integer (int): Postive integer for conversion
        n (int): Number of bits

    Raises:
        OverflowError: Bits should be sufficient to encode integer

    Returns:
        np.ndarray: Bits comprising integer
    """
    if integer >= 2**n:
        raise OverflowError("index requires more than n bits")

    binary_str = format(integer, "b").zfill(n)

    return np.array(list(binary_str), dtype=int)


def integer(bits: np.ndarray) -> int:
    """Convert bit array to integer

    Args:
        bits (np.ndarray): Bit array

    Returns:
        int: Integer encoding of bit array
    """
    bits_str = "".join(str(bit) for bit in bits.flatten())
    integer = int(bits_str, 2)

    return integer


def transformation(n: int) -> np.ndarray:
    """Transformation matrix H (binary to categorical)

    Args:
        n (int): Number of bits

    Returns:
        np.ndarray: n X 2**n matrix with columns encoding ints 0 to 2**n-1
    """
    columns = []
    for i in range(2**n):
        columns.append(binary(i, n))

    return np.vstack(columns).T


def transform(bernoullis: np.ndarray) -> int:
    """Find index for bernoulis as column in H

    Args:
        bernoullis (np.ndarray): Column in H

    Returns:
        int: Index of column in H that matches bernoullis
    """
    return integer(bernoullis)


def aggregate(bernoullis: list[np.ndarray], H: np.ndarray) -> np.ndarray:
    """d vector

    A vector of categorical aggregates for m variables over n trials.

    Args:
        bernoullis (list[np.ndarray]): n X m matrix of Bernoulli variates
        H (np.ndarray): m-dimensional transformation matrix (bit vector to categorical)

    Returns:
        np.ndarray: 2**m entries, each a count of configurations of m variable outomes
    """
    n, m = bernoullis.shape
    w = 2**m
    d = np.zeros((1, w))
    for row in range(n):
        h_index = transform(bernoullis[row][:, None])
        d += np.eye(1, w, h_index)

    return d


class StandardDeviations:
    """Diagonal matrix V of standard deviations"""

    def __init__(self, mu: "Mean", nu: "Shape") -> None:
        """Prior matrix of standard deviations

        Args:
            mu (Mean): Mean object
            nu (Shape): Shape object
        """
        self.mu = mu
        self.nu = nu
        self._data = np.zeros(len(self.mu))
        self.update(self.mu, self.nu)

    def update(self, mu: "Mean", nu: "Shape") -> "StandardDeviations":
        """Posterior matrix of standard deviations

        Args:
            mu (Mean): Mean object
            nu (Shape): Shape object

        Returns:
            StandardDeviations: Standard deviations object
        """
        self._data = np.diag(np.multiply(mu(), 1 - mu()).flatten()) / (nu() + 1)

        return self

    def __call__(self) -> np.ndarray:
        """Value of standard deviations

        Returns:
            np.ndarray: Standard deviations matrix
        """
        return self._data


class Correlation:
    """Correlation matrix R"""

    def __init__(self, m: int) -> None:
        """Prior correlation matrix R

        Args:
            m (int): Number of dimensions
        """
        self.m = m
        self._data = np.eye(self.m)

    def update(self, V: StandardDeviations, Sigma: "Covariance") -> np.ndarray:
        """Posterior matrix R

        Args:
            V (np.ndarray): Standard deviations object
            Sigma (Covariance): Covariance object containing mXm matrix

        Returns:
            Correlation: Correlation object
        """
        V_inv = np.linalg.inv(V())
        self._data = V_inv**0.5 @ Sigma() @ V_inv**0.5

        return self

    def __call__(self) -> np.ndarray:
        """Value of correlation matrix

        Returns:
            np.ndarray: Correlation matrix
        """
        return self._data


class Update:
    """Data matrix U"""

    def __init__(self, bernoullis: list[np.ndarray]):
        self.bernoullis = bernoullis

    def direct(self, H: np.ndarray) -> np.ndarray:
        """Directly compute U from H

        H is an m X 2**m transformation matrix. U is an m X m
        update matrix.

        Args:
            H (np.ndarray): Transformation matrix (bit vector to categorical)

        Returns:
            np.ndarray: Update matrix U
        """
        n, _ = self.bernoullis.shape
        w = H.shape[1]
        Delta = np.zeros((w, w))
        for row in range(n):
            h_index = transform(self.bernoullis[row][:, None])
            Delta[h_index, h_index] += 1

        return H @ Delta @ H.T

    def indirect(self) -> np.ndarray:
        """Calculate U without materializing H

        Returns:
            np.ndarray: Update matrix
        """
        m = self.bernoullis.shape[1]
        U = np.zeros((m, m))
        for row in range(self.bernoullis.shape[0]):
            bits = self.bernoullis[row, :]
            U += np.outer(bits, bits)

        return U


class Shape:
    """Shape parameter nu"""

    def __init__(self, nu: float) -> None:
        """Prior for nu

        Args:
            nu (float): Prior shape parameter
        """
        self._data = nu

    def update(self, n: int) -> "Shape":
        """Posterior for nu

        Args:
            n (int): Number of Bernoulli sample vectors

        Returns:
            Shape: Updated shape object
        """
        self._data += n

        return self

    def __call__(self) -> float:
        """Value of shape

        Returns:
            float: Shape parameter
        """
        return self._data


class Moment:
    """Moment matrix A"""

    def __init__(self, Sigma: "Covariance", mu: "Mean", nu: "Shape") -> None:
        """Prior moment matrix A

        Args:
            Sigma (Covariance): Covariance matrix
            mu (Mean): Mean vector
            nu (Shape): Shape parameter
        """
        self._data = nu() * ((nu() + 1) * Sigma() + np.outer(mu(), mu()))

    def update(self, U: "Update") -> "Moment":
        """Posterior moment matrix

        Args:
            U (Update): Update (data) matrix

        Returns:
            Moment: Updated Moment object
        """
        self._data += U.indirect()

        return self

    def __call__(self) -> np.ndarray:
        """Value of moment matrix

        Returns:
            np.ndarray: Moment matrix
        """
        return self._data


class Mean:
    """Mean vector mu"""

    def __init__(self, m: int) -> None:
        """Prior for mean vector mu

        Args:
            m (int): Number of dimensions of each Bernoulli sample
        """
        self.m = m
        self._data = np.ones((self.m, 1)) / 2

    def update(self, A: "Moment", nu: "Shape") -> "Mean":
        """Posterior for mean vector mu

        Args:
            A (np.ndarray): Moment object
            nu (float): Shape object

        Returns:
            Mean: Updated mean vector
        """
        self._data = np.diag(A()) / nu()

        return self

    def __call__(self) -> np.ndarray:
        """Value of mean vector

        Returns:
            np.ndarray: Mean vector
        """
        return self._data

    def __len__(self) -> int:
        """Length of mean vector

        Returns:
            int: Length
        """
        return max(self._data.shape)


class Covariance:
    """Covariance matrix Sigma"""

    def __init__(self, R: "Correlation", V: "StandardDeviations") -> None:
        """Prior covariance matrix Sigma

        Args:
            R (np.ndarray): Correlation object
            V (np.ndarray): Standard deviations object
        """
        self._data = V() ** 0.5 @ R() @ V() ** 0.5

    def update(self, A: "Moment", mu: "Mean", nu: "Shape") -> "Covariance":
        """Posterior covariance Matrix Sigma

        Args:
            A (Moment): Moment matrix
            mu (Mean): Mean vector
            nu (Shape): Shape parameter

        Returns:
            Covariance: Updated covariance matrix
        """
        self._data = (A() / nu() - np.outer(mu(), mu())) / (nu() + 1)

        return self

    def __call__(self):
        """Value of covariance matrix

        Returns:
            np.ndarray: Covariance matrix
        """
        return self._data


class Marginal:
    """Marginal distribution"""

    def __init__(self, mu: "Mean", nu: "Shape", index: int) -> None:
        """Prior marginal distribution

        Args:
            mu (Mean): Mean object
            nu (Shape): Shape parameter
            index (int): Which marginal
        """
        self.index = index
        self.update(mu, nu)

    def update(self, mu: "Mean", nu: "Shape") -> "Marginal":
        """Posterior update for marginal distribution

        Args:
            mu (Mean): Mean object
            nu (Shape): Shape object

        Returns:
            Marginal: Marginal object
        """
        alpha = nu() * mu()
        beta = nu() * np.ones(alpha.shape) - alpha
        self._dist = stats.beta(alpha[self.index], beta[self.index])

        return self

    def __call__(self) -> stats.beta:
        """Value of marginal distribution

        Returns:
            stats.beta: Marginal distribution
        """
        return self._dist


class mBeta:

    def sample(self, n: int) -> None:
        raise NotImplementedError


class mBetaExact(mBeta):
    """Exact mBeta distribution"""

    def __init__(self, gamma: np.ndarray) -> None:
        """Exact mBeta from gamma parameters

        Args:
            gamma (np.ndarray): Gamma must contain non-zeros
        """
        self.gamma = gamma
        self.m = int(np.log2(len(self.gamma)))
        self.H = transformation(self.m)
        self._dist = stats.dirichlet(self.gamma)

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from exact mBeta distribution

        Args:
            n (int): Number of samples to draw. Defaults to 1.

        Returns:
            np.ndarray: nXlen(gamma) matrix of samples
        """
        return (self.H @ self._dist.rvs(n).T).T


class mBetaApprox(mBeta):
    """Approximate mBeta distribution

    Approximation based on Gaussian copula
    """

    def __init__(self, R: Correlation, mu: Mean, nu: Shape) -> None:
        """Prior approximate mBeta distribution

        Args:
            R (Correlation): Correlation object
            mu (Mean): Mean object
            nu (Shape): Shape object
        """
        self.R = R
        self.mu = mu
        self.nu = nu
        self.m = len(self.mu)
        self.V = StandardDeviations(self.mu, self.nu)
        self.Sigma = Covariance(self.R, self.V)
        self.A = Moment(self.Sigma, self.mu, self.nu)
        self.marginals = [Marginal(self.mu, self.nu, j) for j in range(self.m)]

        self.update()

    def update(self, bernoullis: np.ndarray = None) -> "mBetaApprox":
        """Posterior approximate mBeta distribution

        Args:
            bernoullis (np.ndarray, optional): Observation data. Defaults to None.

        Returns:
            mBeta: mBetaApprox object
        """
        if bernoullis is not None:
            n, _ = bernoullis.shape
            U = Update(bernoullis)
            self.nu.update(n)
            self.A.update(U)
            self.mu.update(self.A, self.nu)
            self.Sigma.update(self.A, self.mu, self.nu)
            self.V.update(self.mu, self.nu)
            self.R.update(self.V, self.Sigma)

            for j in range(self.m):
                self.marginals[j].update(self.mu, self.nu)

        self._dist = CopulaDistribution(
            copula=GaussianCopula(self.R(), k_dim=self.m, allow_singular=False),
            marginals=[marginal() for marginal in self.marginals],
        )

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from approximate mBeta distribution

        Args:
            n (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: nXm matrix of samples
        """
        if n == 1:
            # there is a bug where 1 sample throws a dimension mismatch error
            return self._dist.rvs(2)[0, :]
        else:
            return self._dist.rvs(n)

    def __call__(self) -> CopulaDistribution:
        """Value of approximate mBeta

        Returns:
            CopulaDistribution: Approximate mBeta distribution
        """
        return self._dist


class mBetaMixture(mBeta):
    """A mixture of mBeta distributions"""

    def __init__(self, mBetas: list[mBeta]) -> None:
        """A collection of mBetas

        Args:
            mBetas (list[mBeta]): Component mBeta objects
        """
        self.mBetas = mBetas

    def sample(self, n: int = 1) -> np.ndarray:
        """Choose an mBeta uniformly at random, then sample from it

        Args:
            n (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            np.ndarray: Samples from mBetaMixture
        """
        samples = []
        for i in range(n):
            mBeta = random.choice(self.mBetas)
            samples.append(mBeta.sample())

        return np.vstack(samples)
