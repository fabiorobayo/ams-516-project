import numpy as np
from pomegranate import NormalDistribution
from hmmlearn import hmm
from scipy import optimize


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


def log_likelihood_discrete(delta, Gamma, pi, observations, weights=None):
    # Calculate log-likelihood using methodology described in the paper below for the discrete case
    # "Exact Computation of the Observed Information Matrix for Hidden Markov Models" (Lystig, T; Hughes, J)

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

    l = np.log(Lambda).dot(weights)

    return l


def log_likelihood(delta, Gamma, pi, observations, weights=None):
    # Calculate log-likelihood using methodology described in the paper below for the continuous case
    # "Exact Computation of the Observed Information Matrix for Hidden Markov Models" (Lystig, T; Hughes, J)

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

    l = np.log(Lambda).dot(weights)

    return l


def params_to_vec(delta, Gamma, pi):
    return  np.array([[delta[0]], [Gamma[0, 0]], [Gamma[1, 1]], [pi[0].mu], [pi[0].sigma], [pi[1].mu], [pi[1].sigma]], dtype=object)


def vec_to_params(x):
    x = x.squeeze()
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
    deltaGrad[0, 0] = -1
    deltaGrad[1, 0] = 1

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

        piGrad[i, piStart + i * numStates] = pi[i].density_d_mu
        piGrad[i, piStart + i * numStates + 1] = pi[i].density_d_sigma

    return deltaGrad, GammaGrad, piGrad


def get_model_hessians(numStates, pi):
    numParams = (numStates - 1) + numStates * (numStates - 1) + numStates * 2

    deltaHess = np.zeros((numStates, numParams, numParams))
    GammaHess = np.zeros((numStates, numStates, numParams, numParams))
    piHess = np.zeros((numStates, numParams, numParams), dtype=object)

    # The code below assumes 2 states but can be modified to more states
    # deltaGrad and GammaGrad are constant so the hessians are zero

    piStart = (numStates - 1) + numStates * (numStates - 1)
    for i in range(numStates):

        piHess[i, piStart + i * numStates, piStart + i * numStates] = pi[
            i
        ].density_d_mu_2
        piHess[i, piStart + i * numStates + 1, piStart + i * numStates] = pi[
            i
        ].density_d_mu_d_sigma
        piHess[i, piStart + i * numStates, piStart + i * numStates + 1] = pi[
            i
        ].density_d_mu_d_sigma
        piHess[i, piStart + i * numStates + 1, piStart + i * numStates + 1] = pi[
            i
        ].density_d_sigma_2

    return deltaHess, GammaHess, piHess

def calculate_score(delta, Gamma, pi, observations, weights=None):
    # Calculate score following methodology in
    # "Exact Computation of the Observed Information Matrix for Hidden Markov Models" (Lystig, T; Hughes, J)

    s = len(delta)
    T = len(observations)

    deltaGrad, GammaGrad, piGrad = get_model_gradients(s, pi)

    numParams = deltaGrad.shape[1]

    if weights is None:
        weights = np.ones(T)

    _lambda = np.zeros((s, T))
    Lambda = np.zeros(T)
    psi = np.zeros((s, T, numParams))
    Psi = np.zeros((T, numParams))

    # t = 0
    for j in range(s):
        _lambda[j, 0] = pi[j].probability(observations[0]) * delta[j]
        for p in range(numParams):
            if callable(piGrad[j, p]):
                dPi = piGrad[j, p](observations[0])
            else:
                dPi = piGrad[j, p]
            # score calculation
            psi[j, 0, p] = (
                dPi * delta[j] + pi[j].probability(observations[0]) * deltaGrad[j, p]
            )


    Lambda[0] = _lambda[:, 0].sum()
    Psi[0, :] = weights[0] * psi[:, 0, p].sum(axis=0)

    # t > 0
    for t in range(1, T):
        for j in range(s):
            pitj = pi[j].probability(observations[t])
            for i in range(s):
                gammaij = Gamma[i, j]
                _lambda[j, t] += _lambda[i, t - 1] * pitj * gammaij
                for p in range(numParams):
                    if callable(piGrad[j, p]):
                        dPi = piGrad[j, p](observations[t])
                    else:
                        dPi = piGrad[j, p]
                    # score calculation
                    psi[j, t, p] += (
                        psi[i, t - 1, p] * pitj * gammaij
                        + _lambda[i, t - 1] * dPi * gammaij
                        + _lambda[i, t - 1] * pitj * GammaGrad[i, j, p]
                    )

            _lambda[j, t] /= Lambda[t - 1]
            psi[j, t, :] /= Lambda[t - 1]

        Lambda[t] = _lambda[:, t].sum()
        Psi[t, :] = weights[t] * psi[:, t, :].sum(axis=0)

    l = np.log(Lambda).dot(weights)
    score = Psi[T - 1, :] / Lambda[T - 1]

    return l, score[:, np.newaxis]


def estimate_weighted_score(theta_0, T, log_returns, f):
    l = 0
    deltaHat, GammaHat, piHat = vec_to_params(theta_0)

    for t in range(1, T):
        w = f ** (T - t)
        l_t, score_t = calculate_score(deltaHat, GammaHat, piHat, log_returns[:t])
        l += w * l_t

        if t > 1:
            score = score + (score_t - score) * w
            inf = inf + (1 / t) * (score @ score.T - inf)
        else:
            score = score_t
            inf = score @ score.T

    return score, inf, l


def score_and_information(delta, Gamma, pi, observations, weights=None):
    # Calculate score and information matrix following methodology in
    # "Exact Computation of the Observed Information Matrix for Hidden Markov Models" (Lystig, T; Hughes, J)

    s = len(delta)
    T = len(observations)

    deltaGrad, GammaGrad, piGrad = get_model_gradients(s, pi)
    deltaHessian, GammaHessian, piHessian = get_model_hessians(s, pi)

    numParams = deltaGrad.shape[1]

    if weights is None:
        weights = np.ones(T)

    _lambda = np.zeros((s, T))
    Lambda = np.zeros(T)
    psi = np.zeros((s, T, numParams))
    Psi = np.zeros((T, numParams))
    omega = np.zeros((s, T, numParams, numParams))
    Omega = np.zeros((T, numParams, numParams))

    # t = 0
    for j in range(s):
        _lambda[j, 0] = pi[j].probability(observations[0]) * delta[j]
        for p in range(numParams):
            if callable(piGrad[j, p]):
                dPi = piGrad[j, p](observations[0])
            else:
                dPi = piGrad[j, p]
            # score calculation
            psi[j, 0, p] = (
                dPi * delta[j] + pi[j].probability(observations[0]) * deltaGrad[j, p]
            )

            for q in range(numParams):
                if callable(piGrad[j, q]):
                    dqPi = piGrad[j, q](observations[0])
                else:
                    dqPi = piGrad[j, q]

                if callable(piHessian[j, p, q]):
                    d2Pi = piHessian[j, p, q](observations[0])
                else:
                    d2Pi = piHessian[j, p, q]
                # Information matrix calculation. The fourth term in the sum is really always zero,
                # but we leave it for consistency with the paper
                omega[j, 0, p, q] = (
                    d2Pi * delta[j]
                    + dPi * deltaGrad[j, q]
                    + dqPi * deltaGrad[j, p]
                    + pi[j].probability(observations[0]) * deltaHessian[j, p, q]
                )

    Lambda[0] = _lambda[:, 0].sum()
    Psi[0, :] = weights[0] * psi[:, 0, p].sum(axis=0)
    Omega[0, :, :] = weights[0] * omega[:, 0, :, :].sum(axis=0)

    # t > 0
    for t in range(1, T):
        for j in range(s):
            pitj = pi[j].probability(observations[t])
            for i in range(s):
                gammaij = Gamma[i, j]
                _lambda[j, t] += _lambda[i, t - 1] * pitj * gammaij
                for p in range(numParams):
                    if callable(piGrad[j, p]):
                        dPi = piGrad[j, p](observations[t])
                    else:
                        dPi = piGrad[j, p]
                    # score calculation
                    psi[j, t, p] += (
                        psi[i, t - 1, p] * pitj * gammaij
                        + _lambda[i, t - 1] * dPi * gammaij
                        + _lambda[i, t - 1] * pitj * GammaGrad[i, j, p]
                    )

                    for q in range(numParams):
                        if callable(piGrad[j, q]):
                            dqPi = piGrad[j, q](observations[t])
                        else:
                            dqPi = piGrad[j, q]

                        if callable(piHessian[j, p, q]):
                            d2Pi = piHessian[j, p, q](observations[t])
                        else:
                            d2Pi = piHessian[j, p, q]
                        # Information matrix calculation. The ninth term in the sum is really always zero,
                        # but we leave it for consistency with the paper
                        omega[j, t, p, q] += (
                            omega[i, t - 1, p, q] * pitj * gammaij
                            + psi[i, t - 1, p] * dqPi * gammaij
                            + psi[i, t - 1, p] * pitj * GammaGrad[i, j, q]
                            + psi[i, t - 1, p] * dPi * gammaij
                            + psi[i, t - 1, p] * pitj * GammaGrad[i, j, p]
                            + _lambda[i, t - 1] * d2Pi * gammaij
                            + _lambda[i, t - 1] * dPi * GammaGrad[i, j, q]
                            + _lambda[i, t - 1] * dqPi * GammaGrad[i, j, p]
                            + _lambda[i, t - 1] * pitj * GammaHessian[i, j, p, q]
                        )

            _lambda[j, t] /= Lambda[t - 1]
            psi[j, t, :] /= Lambda[t - 1]
            omega[j, t, :, :] /= Lambda[t - 1]

        Lambda[t] = _lambda[:, t].sum()
        Psi[t, :] = weights[t] * psi[:, t, :].sum(axis=0)
        Omega[t, :, :] = weights[t] * omega[:, t, :, :].sum(axis=0)

    l = np.log(Lambda).dot(weights)
    score = Psi[T - 1, :] / Lambda[T - 1]
    information = (
        -Omega[T - 1, :, :] / Lambda[T - 1]
        + (Psi[T - 1][:, np.newaxis] @ Psi[T - 1][np.newaxis, :]) / Lambda[T - 1] ** 2
    )

    return l, score[:, np.newaxis], information


def get_hmm_forecasts(K, delta, Gamma, pi, log_returns):

    numStates = len(delta)
    alpha = delta @ np.eye(numStates)
    for y_t in log_returns:
        P_t = np.diag([pi[0].probability(y_t), pi[1].probability(y_t)])
        alpha = alpha @ (Gamma @ P_t)
        alpha = alpha / alpha.sum()


    alphas = np.zeros((K, numStates))
    alphas[0] = alpha.T
    for i in range(1, K):
        alphas[i] = alphas[i - 1] @ Gamma

    means = np.array([[pi[0].mu], [pi[1].mu]])
    sigmas = np.array([[pi[0].sigma], [pi[1].sigma]])

    mu = alphas @ means

    sigma_sqd = alphas @ (means ** 2 + sigmas ** 2) - mu ** 2

    forecasted_mean = np.exp(mu + sigma_sqd / 2) - 1
    forecasted_var = (np.exp(sigma_sqd) - 1) * np.exp(2 * mu + sigma_sqd)

    return forecasted_mean, forecasted_var


def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha


def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta


def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return {"a": a, "b": b}


def initialize_theta(start, theta_0, log_returns, T, f, A):

    l = 0
    deltaHat, GammaHat, piHat = vec_to_params(theta_0)
    thetas = [theta_0.squeeze()]

    for t in range(1, T):
        w = f ** (T - t)
        l_t, score_t, inf_t = score_and_information(deltaHat, GammaHat, piHat, log_returns[:t])
        # l_t, score_t = calculate_score(deltaHat, GammaHat, piHat, log_returns[:t])
        l += w * l_t

        if t > 1:
            score = score + (score_t - score) * w
            # inf = inf + (1 / t) * (score @ score.T - inf)
            inf = inf + (inf_t - inf) * w
        else:
            score = score_t
            # inf = score @ score.T
            inf = inf_t
            theta_hat = theta_0

        if t > start:
            theta_hat = theta_hat + A * np.linalg.inv(inf) @ score
            thetas.append(theta_hat.squeeze())
            if np.isnan(theta_hat.astype(float)).any() or (theta_hat[4] < 0) or (theta_hat[6] < 0):
                print('NaN found')
            theta_hat[0] = min(max(theta_hat[0], 0), 1)
            theta_hat[1] = min(max(theta_hat[1], 0), 1)
            theta_hat[2] = min(max(theta_hat[2], 0), 1)
            theta_hat[3] = min(max(theta_hat[3], -1 / 252), 1 / 252)
            theta_hat[4] = min(max(theta_hat[4], 0.00001), 1 / np.sqrt(252))
            theta_hat[5] = min(max(theta_hat[5], -1 / 252), 1 / 252)
            theta_hat[6] = min(max(theta_hat[6], 0.00001), 1 / np.sqrt(252))
            deltaHat, GammaHat, piHat = vec_to_params(theta_hat.squeeze())

    return theta_hat


def estimate_parameters(observations, method='em', theta_0=None, memLength=None):

    numObservations = len(observations)
    if memLength:
        f = 1 - 1 / memLength
        A = 1 / memLength
        weights = f**np.arange(numObservations, 0, -1)
    else:
        weights = np.ones(numObservations)/numObservations

    if theta_0 is None:
        delta_r = np.random.randn(1) * 0.01 + (1 / 2)
        delta = np.array([delta_r[0], 1 - delta_r[0]])
        r = np.random.randn(2, 1) * 0.01 + (1 / 2)
        Gamma = np.hstack([r, 1 - r])
        pi_params = np.array([[0.17 / 252, 0.00104 / np.sqrt(252)], [-0.3 / 252, 0.00104 / np.sqrt(252)]])
        pi = [NDist(p[0], p[1]) for p in pi_params]
    else:
        delta, Gamma, pi = vec_to_params(theta_0)
        pi_params = np.array([[p.mu,p.sigma] for p in pi])


    if method == 'em':
        model = hmm.GaussianHMM(n_components=2, covariance_type="spherical", n_iter=100, tol=0.000001)
        model.startprob_prior_ = delta
        model.transmat_prior = Gamma
        model.means_prior = pi_params[:, 0][:, np.newaxis]
        model.covars_prior = np.array([[pi_params[0, 1]], [pi_params[1, 1]]])
        model.fit(observations)
        pi = [NDist(model.means_[0][0],np.sqrt(model.covars_[0][0][0][0])),NDist(model.means_[1][0],np.sqrt(model.covars_[1][0][0][0]))]
        delta = model.startprob_
        Gamma = model.transmat_

    elif method == 'newton':
        if theta_0 is None:
            start = 50
            N_eff = 260
            f = 1 - 1 / N_eff
            A = 1 / N_eff
            pi_params = np.array([[0.17 / 252, 0.11 / np.sqrt(252)], [-0.32 / 252, 0.35 / np.sqrt(252)]])
            Gamma = np.array([[0.99, 0.01], [0.035, 0.965]])
            delta = np.array([0.99, 0.01])
            pi = [NDist(p[0], p[1]) for p in pi_params]
            theta_0 = params_to_vec(delta, Gamma, pi)
            theta_hat = initialize_theta(start, theta_0, observations, N_eff, f, A)
            delta, Gamma, pi = vec_to_params(theta_hat)
        else:
            theta_hat = theta_0
            delta_hat, Gamma_hat, pi_hat = vec_to_params(theta_hat)
            l_hat, score_hat, inf_hat = score_and_information(delta_hat, Gamma_hat, pi_hat, observations, weights)
            try:
                theta_hat = theta_hat + A * np.linalg.inv(inf_hat) @ score_hat
            except np.linalg.LinAlgError:
                print('Singular Matrix')
                pass

            if np.isnan(theta_hat.astype(float)).any() or (theta_hat[4] < 0) or (theta_hat[6] < 0):
                print('NaN found')
            theta_hat[0] = min(max(theta_hat[0], 0), 1)
            theta_hat[1] = min(max(theta_hat[1], 0), 1)
            theta_hat[2] = min(max(theta_hat[2], 0), 1)
            theta_hat[3] = min(max(theta_hat[3], -1 / 252), 1 / 252)
            theta_hat[4] = min(max(theta_hat[4], 0.00001), 1 / np.sqrt(252))
            theta_hat[5] = min(max(theta_hat[5], -1 / 252), 1 / 252)
            theta_hat[6] = min(max(theta_hat[6], 0.00001), 1 / np.sqrt(252))
            delta, Gamma, pi = vec_to_params(theta_hat)

    elif method == "num-opt":
        foo = lambda x: -log_likelihood_optim(x, observations, weights)
        if theta_0 is None:
            theta_0 = params_to_vec(delta, Gamma, pi)

        result_nm = optimize.minimize(
            foo,
            np.array(theta_0),
            method='L-BFGS-B',
            bounds=[
                (0, 0.99),
                (0, 0.99),
                (0, 0.99),
                (-0.05, 0.05),
                (0.002, 0.05),
                (-0.1, 0.1),
                (0.002, 0.05)
            ])
        delta, Gamma, pi = vec_to_params(result_nm.x)
    else:
        raise ValueError(f'method {method} not defined')

    if np.isnan(delta).any() or np.isnan(Gamma).any():
        # raise ValueError('NaN found in HMM parameters')
        print('NaN found in HMM parameters')

    return delta, Gamma, pi