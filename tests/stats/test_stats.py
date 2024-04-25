import numpy as np
import scipy
import statsmodels

from fair.stats import (
    Correlation,
    Covariance,
    Marginal,
    Mean,
    Moment,
    Shape,
    StandardDeviations,
    Update,
    aggregate,
    binary,
    integer,
    mBetaApprox,
    mBetaExact,
    transformation,
)


def test_index_to_vector():
    np.testing.assert_array_equal(binary(3, 2), np.array([1, 1]))
    np.testing.assert_array_equal(binary(3, 3), np.array([0, 1, 1]))
    with np.testing.assert_raises(OverflowError):
        binary(3, 1)


def test_convert_int_bits():
    assert integer(binary(5, 3)) == 5
    assert integer(binary(13, 4)) == 13
    assert integer(binary(0, 2)) == 0


def test_transform():
    bits = np.array([1, 0, 1]).reshape((3, 1))
    H3 = transformation(3)
    index = np.where((H3 == bits).all(axis=0))[0][0]

    assert index == integer(bits)


def test_transformation():
    trans = transformation(3)
    H3 = np.array(
        [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]]
    )

    np.testing.assert_array_equal(trans, H3)


def test_update(bernoullis: np.ndarray):
    U = Update(bernoullis)

    np.testing.assert_array_equal(U.direct(transformation(3)), U.indirect())


def test_prior_posterior(bernoullis: np.ndarray):
    # data
    U = Update(bernoullis)
    n, m = bernoullis.shape

    # priors
    R = Correlation(m)
    nu = Shape(1)
    mu = Mean(m)
    V = StandardDeviations(mu, nu)
    Sigma = Covariance(R, V)
    A = Moment(Sigma, mu, nu)

    # posteriors
    nu.update(n)
    A.update(U)
    mu.update(A, nu)
    Sigma.update(A, mu, nu)
    V.update(mu, nu)
    R.update(V, Sigma)


def test_marginal(bernoullis: np.ndarray):
    n, m = bernoullis.shape
    # which dimension to use for marginal
    j = 1
    marginal = Marginal(Shape(1), Mean(m), 0)
    marginal.update(Shape(2), Mean(m))

    assert isinstance(
        marginal(), scipy.stats._distn_infrastructure.rv_continuous_frozen
    )


def test_exact_mbeta():
    m = 3
    eps = 0.01
    gamma = np.ones((2**m,)) / eps
    gamma[1] = 10
    gamma[5] = 1
    mbeta = mBetaExact(gamma)

    assert mbeta.sample(2).shape == (2, m)


def test_mbeta(bernoullis: np.ndarray):
    _, m = bernoullis.shape
    R = Correlation(m)
    nu = Shape(1)
    mu = Mean(m)
    mbeta = mBetaApprox(R, mu, nu)
    mbeta.update(bernoullis)

    assert isinstance(mbeta(), statsmodels.distributions.copula.api.CopulaDistribution)
    assert len(mbeta.sample()) == m
    assert mbeta.sample(2).shape == (2, m)


def test_aggregates_not_enough_for_U():
    """Survey aggregates are not enough to form update matrix U

    This test provides a counterexample to the idea that, for the purpose of computing
    update matrix U, one can treat the surveyed values, 0-n for each of m courses, as n
    *simultaneous* draws from m Bernoulli distributions. The counterexample shows that
    two different sets of n flips will, having the same aggregates, will produce two
    different update matrices.
    """
    H3 = transformation(3)

    # example survey outcomes; each outcome has n=2 sets of m=3 coin flips
    bernoullis1 = np.array([[1, 1, 1], [0, 1, 0]])
    bernoullis2 = np.array([[1, 1, 0], [0, 1, 1]])
    w = 2 ** bernoullis1.shape[1]

    # aggregates of the flips are equal
    np.testing.assert_array_equal(
        np.sum(bernoullis1, axis=0), np.sum(bernoullis2, axis=0)
    )

    # form d vectors: sum of basis vectors corresponding to column indices of H3 that match
    # rows of the outcomes vectors
    d_v1 = aggregate(bernoullis1, H3)
    d_v2 = aggregate(bernoullis2, H3)

    # form Delta matrices: diagonal matrix formed from d
    # pre- and post-multiply Delta by H3 to form U matrices
    U_v1 = H3 @ np.diag(d_v1.reshape((w,))) @ H3.T
    U_v2 = H3 @ np.diag(d_v2.reshape((w,))) @ H3.T

    # show that the two U matrices differ
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(U_v1, U_v2)
