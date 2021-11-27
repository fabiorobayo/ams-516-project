import numpy as np
from pomegranate import NormalDistribution


class NDist(NormalDistribution):
    def __init__(self, mean, sigma):
        super().__init__(mean, sigma)
        self.mu = mean
        self.sigma = sigma

    def density_d_mu(self, x):
        return self.probability(x) * (x - self.mu) / (self.sigma ** 2)

    def density_d_sigma(self, x):
        y = (x - self.mu) / self.sigma
        return self.probability(x) * (y ** 2 - 1) / self.sigma

    def density_d_mu_2(self, x):
        y = (x - self.mu) / self.sigma
        return self.probability(x) * (y ** 2 - 1) / (self.sigma ** 2)

    def density_d_mu_d_sigma(self, x):
        y = (x - self.mu) / self.sigma
        return self.probability(x) * (y ** 2 - 3) * (x - self.mu) / (self.sigma ** 3)

    def density_d_sigma_2(self, x):
        y = (x - self.mu) / self.sigma
        return self.probability(x) * (pow(y, 4) - y ** 2 + 2) / (self.sigma ** 2)


def log_likelihood_discrete(delta, Gamma, pi, observations, weights):

    s = len(delta)
    T = len(observations)

    if weights is None:
        weights = np.ones(T)

    _lambda = np.zeros((s, T))
    Lambda = np.zeros(T)

    for j in range(s):
        _lambda[j, 0] = pi[j, observations[0]] * delta[j]

    Lambda[0] = _lambda[:, 0].sum()
    for t in range(1, T):
        for j in range(s):
            for i in range(s):
                _lambda[j, t] += (
                    _lambda[i, t - 1] * pi[j, observations[t]] * Gamma[i, j]
                )
            _lambda[j, t] /= Lambda[t - 1]
        Lambda[t] = _lambda[:, t].sum()

    l = np.log(Lambda.dot(weights))

    return l


def log_likelihood(delta, Gamma, pi, observations, weights=None):

    s = len(delta)
    T = len(observations)

    if weights is None:
        weights = np.ones(T)

    _lambda = np.zeros((s, T))
    Lambda = np.zeros(T)

    for j in range(s):
        _lambda[j, 0] = pi[j].probability(observations[0]) * delta[j]

    Lambda[0] = _lambda[:, 0].sum()
    for t in range(1, T):
        for j in range(s):
            for i in range(s):
                _lambda[j, t] += (
                    _lambda[i, t - 1] * pi[j].probability(observations[t]) * Gamma[i, j]
                )
            _lambda[j, t] /= Lambda[t - 1]
        Lambda[t] = _lambda[:, t].sum()

    l = np.log(Lambda.dot(weights))

    return l


def vec_to_params(x):
    delta = np.array([x[0], 1 - x[0]])
    Gamma = np.array([[x[1], 1 - x[1]], [1 - x[2], x[2]]])
    pi = [NDist(x[3], x[4]), NDist(x[5], x[6])]

    return delta, Gamma, pi


def log_likelihood_optim(x, observations, weights=None):

    delta, Gamma, pi = vec_to_params(x)

    return log_likelihood(delta, Gamma, pi, observations, weights)


def get_model_gradients(numStates, pi):

    numParams = (numStates - 1) + numStates * (numStates - 1) + numStates * 2

    deltaGrad = np.zeros((numStates, numParams))
    GammaGrad = np.zeros((numStates, numStates, numParams))
    piGrad = np.zeros((numStates, numParams), dtype=object)

    # The code below assumes 2 states but can be modified to more states
    # d p1 / d p1  = 1, d (1 - p1) / d p1 = -1
    deltaGrad[0, :numStates] = 1
    deltaGrad[1, :numStates] = -1

    # d gamma_11 / d gamma_11 = 1
    # d gamma_12 = (1 - gamma_11)/ d gamma_11 = -1
    # d gamma_21 / d gamma_11 = 0
    # d gamma_22 / d gamma_11 = 0
    GammaGrad[:, :, 1] = np.array([[1, -1], [0, 0]])

    # d gamma_11 / d gamma_22 = 0
    # d gamma_12 / d gamma_22 = 0
    # d gamma_21 = (1 - gamma_22) / d gamma_22 = -1
    # d gamma_22 / d gamma_22 = 1
    GammaGrad[:, :, 2] = np.array([[0, 0], [-1, 1]])

    piStart = (numStates - 1) + numStates * (numStates - 1)
    for i in range(numStates):

        piGrad[i, piStart + i * (numStates)] = pi[i].density_d_mu
        piGrad[i, piStart + i * (numStates) + 1] = pi[i].density_d_sigma

    return deltaGrad, GammaGrad, piGrad


def get_model_hessians(numStates, pi):
    # WORK IN PROGRESS
    pass


def score_and_information(delta, Gamma, pi, observations, weights=None):
    # WORK IN PROGRESS

    s = len(delta)
    T = len(observations)

    deltaGrad, GammaGrad, piGrad = get_model_gradients(s, pi)
    # deltaHessian, GammaHessian, piHessian = get_model_hessians(s, pi)

    numParams = len(deltaGrad)

    if weights is None:
        weights = np.ones(T)

    _lambda = np.zeros((s, T))
    Lambda = np.zeros(T)
    psi = np.zeros((s, T, numParams))
    Psi = np.zeros(T, numParams)
    omega = np.zeros((s, T, numParams, numParams))
    Omega = np.zeros((T, numParams, numParams))

    for j in range(s):

        _lambda[j, 0] = pi[j, observations[0]] * delta[j]

    Lambda[0] = _lambda[:, 0].sum()
    for t in range(1, T):
        for j in range(s):
            for i in range(s):
                _lambda[j, t] += (
                    _lambda[i, t - 1] * pi[j, observations[t]] * Gamma[i, j]
                )
            _lambda[j, t] /= Lambda[t - 1]
        Lambda[t] = _lambda[:, t].sum()

    l = np.log(Lambda.dot(weights))

    return l


def get_hmm_forecasts(K, delta, Gamma, pi, log_returns):

    numStates = len(delta)
    Pi = np.eye(numStates)
    for y_t in log_returns[:K]:
        P_t = np.diag([pi[0].probability(y_t), pi[1].probability(y_t)])
        Pi = Pi @ (Gamma @ P_t)

    alpha = (delta.T @ Pi / ((delta.T @ Pi).sum())).T

    alphas = np.zeros((K, numStates))
    alphas[0] = alpha.T
    for i in range(1, K):
        alphas[i] = alphas[i - 1] @ Gamma

    means = np.array([[pi[0].mu], [pi[1].mu]])
    sigmas = np.array([[pi[0].mu], [pi[1].sigma]])

    mu = alphas @ means

    sigma_sqd = alphas @ (means ** 2 + sigmas ** 2) + mu ** 2

    forecasted_mean = np.exp(mu + sigma_sqd / 2) - 1
    forecasted_var = (np.exp(sigma_sqd) - 1) * np.exp(2 * mu + sigma_sqd)

    return forecasted_mean, forecasted_var