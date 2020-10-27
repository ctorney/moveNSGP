
# Copyright 2019 Colin Torney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapative Truncated Metropolis Adjusted Langevin Algorithm 
# code modified version of https://github.com/gaspardbb/Truncated-MALA


import numpy as np
from tqdm import tqdm


def log_normal_pdf_unn(x, mean, variance, inv_variance=None):
    """
    log of unnormalized pdf of a normal distribution.

    Parameters
    ----------
    x
        Where to evaluate the Normal.
    mean, variance:
        parameters of the gaussian.
    inv_variance
        Inverse of the variance (to avoid computing it again and again in certain cases).

    Returns
    -------
    float: the result.
    """
    if inv_variance is None:
        inv_variance = np.linalg.inv(variance)
    return -1 / 2 * (x - mean) @ (inv_variance @ (x - mean))





def projection_operators(epsilon_1, A_1):
    """
    Return the projection functions defined in the article.

    Parameters
    ----------
    epsilon_1, A_1:
        The two scaling operators.

    Returns
    -------
    proj_sigma: Projection on segment epsilon_1, A_1
    proj_gamma: Projection on cone of definite matrix of norm < 1_1
    proj_mu: Projection on centered ball of radius A_1
    """

    def proj_sigma(x: float) -> float:
        if x < epsilon_1:
            return epsilon_1
        elif x > A_1:
            return A_1
        else:
            return x

    def proj_gamma(x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2
        norm = np.linalg.norm(x, ord='fro')  # Frobenius norm
        if norm > A_1:
            print(f"Projection on Gamma! norm={norm:.1e}")
            return A_1 / norm * x
        else:
            return x

    def proj_mu(x: np.ndarray) -> np.ndarray:
        assert x.ndim == 1
        norm = np.linalg.norm(x, ord=2)
        if norm > A_1:
            print(f"Projection on mu! norm={norm:.1e}")
            return A_1 / norm * x
        else:
            return x

    return proj_sigma, proj_gamma, proj_mu



    


class AdaptiveMALA():
    def __init__(self, state, log_pdf_and_drift,
                 epsilon_1=1e-5,
                 epsilon_2=1e-6,
                 A_1=1e7,
                 tau_bar=0.574,
                 mu_0=None,
                 gamma_0=None,
                 sigma_0=1,
                 robbins_monroe=10,
                 threshold_start_estimate=100,
                 threshold_use_estimate=200,
                 ):
        """
        Adaptative MALA sampler, described in [1].

        Parameters
        ----------
        state: initial state to start in.
        pi: Callable. Unnormalized pdf of the distribution we want to approximate.
        log_pi: log of the distribution we want to approximate
        drift: Callable.
        epsilon_1, epsilon_2, A_1: parameters of the HM algorithm. Must verify: 0 < epsilon_1 < A_1, 0 < epsilon_2.
        tau_bar: target optimal acceptation rate.
        mu_0, gamma_0, sigma_0: initial values for the parameters.
        robbins_monroe: constant c_0 for the robbins monroe coefficients: g_n = c_0/n
        threshold_use_estimate: int corresponding to the number of steps after which we start updating the covariance
        matrix

        References
        ----------
        [1] An adaptive version for the Metropolis adjusted Langevin algorithm with a truncated drift, Yves F. Atchadé

        """
        
        self.dims = state.shape[0]
        self.state = state
        self.log_pdf_and_drift = log_pdf_and_drift
        self.acceptance_rate = 0
        self.steps = 0
        self.history = {'state': [state], 'acceptance rate': []}

        if mu_0 is None:
            mu_0 = state
        if gamma_0 is None:
            gamma_0 = np.eye(self.dims)

        self.tau_bar = tau_bar
        self.gamma = gamma_0
        self.sigma = sigma_0
        self.A_1 = A_1
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.c_0 = robbins_monroe
        self._gamma_estimate = self.gamma.copy()
        self.params_history = {'gamma': [gamma_0.copy()],'sigma': [sigma_0]}


        self.epsilon_1 = epsilon_1

        self.mu = mu_0
        self.proj_sigma, self.proj_gamma, self.proj_mu = projection_operators(epsilon_1, A_1)
        self.threshold_use_estimate = threshold_use_estimate
        self.threshold_start_estimate = threshold_start_estimate
        self.params_history['mu'] = [mu_0.copy()]

    
    def run_sampler(self,n_samples, n_burn, thin=1):
        """
        Draw samples from the chain.

        Returns
        -------
        The new state.
        """
        self.steps = 0
        
        samples = np.zeros((n_samples,self.dims))
        
        # total iterations for number of samples
        iters = (n_samples * thin) + n_burn
        
        
        pbar = tqdm(range(iters))
        self.state_log_pdf, self.state_drift = self.log_pdf_and_drift(self.state)

        for i in pbar:
            self.steps += 1
            self.proposal = self.proposal_sampler()
            self.proposal_log_pdf, self.proposal_drift = self.log_pdf_and_drift(self.proposal)

            alpha = self.acceptance_ratio()
            
            u = np.random.uniform(0, 1)
            if u <= alpha:
                self.state = self.proposal
                self.acceptance_rate = ((self.steps - 1) * self.acceptance_rate + 1) / self.steps
                self.state_log_pdf = self.proposal_log_pdf.copy()
                self.state_drift = self.proposal_drift.copy()

            else:
                self.acceptance_rate = ((self.steps - 1) * self.acceptance_rate) / self.steps
            pbar.set_description("AR %f SS %f" % (self.acceptance_rate,self.sigma))

            self.history['state'].append(self.state.copy())
            self.history['acceptance rate'].append(self.acceptance_rate)

            #print(self.state[0])
            if i >= n_burn and i % thin == 0:
                samples[(i - n_burn) // thin] = self.state

            if i < n_burn:
                self.update_params(alpha=alpha)
            assert np.isfinite(self.state).all()
            assert np.isfinite(self.gamma).all()

        return samples
    

    
    def acceptance_ratio(self):
        """
        Compute the alpha parameter for the proposal, given the state we are in (self.state).

        Parameters
        ----------
        proposal
            The proposal value, given by e.g. proposal_sampler()
        log
            Computing acceptance_ratio using log trick

        Returns
        -------
        A float between 0 and 1.
        """
        
        arg_exp = self.proposal_log_pdf - self.state_log_pdf \
                  + self.fwd_log_proposal_value() - self.bkwd_log_proposal_value()
        #arg_exp = arg_exp.numpy()
        if np.isfinite(arg_exp):
            alpha = np.exp(min(0, arg_exp))
        else:
            alpha=0
        
        return alpha
    
    def proposal_sampler(self) -> np.ndarray:
        big_lambda = self.gamma + self.epsilon_2 * np.eye(self.dims)
        mean = self.state + self.sigma ** 2 / 2 * big_lambda @ self.state_drift
        variance = self.sigma ** 2 * big_lambda
        sample = np.random.multivariate_normal(mean=mean, cov=variance)
        return sample

    def fwd_log_proposal_value(self):
        big_lambda = self.gamma + self.epsilon_2 * np.eye(self.dims)
        mean = self.proposal + self.sigma ** 2 / 2 * big_lambda @ self.proposal_drift
        variance = self.sigma ** 2 * big_lambda
        value = log_normal_pdf_unn(self.state, mean, variance)
        return value

    def bkwd_log_proposal_value(self):
        big_lambda = self.gamma + self.epsilon_2 * np.eye(self.dims)
        mean = self.state + self.sigma ** 2 / 2 * big_lambda @ self.state_drift
        variance = self.sigma ** 2 * big_lambda
        value = log_normal_pdf_unn(self.proposal, mean, variance)
        return value
    


    def update_params(self, alpha):
        
        """


        Parameters
        ----------
        model: instance of AdaptiveMALA or AdaptiveSymmetricRW
        alpha: the acceptance ratio

        Updates the parameters of the instance.
        """
        coeff = self.c_0 / self.steps

        


        # _gamma_estimate holds the estimation of the covariance matrix.
        # It is different from gamma: indeed, we want to estimate the covariance matrix without using it at first.
        if self.steps > self.threshold_start_estimate:
            coeff_gamma = self.c_0 / (self.steps - self.threshold_start_estimate)
            covariance = (self.state - self.mu)[:, np.newaxis] @ (self.state - self.mu)[np.newaxis, :]
            #covariance = np.diag((self.state - self.mu)**2)

            self._gamma_estimate = self.proj_gamma(self._gamma_estimate + coeff_gamma * (covariance - self._gamma_estimate))
        if self.steps > self.threshold_use_estimate:
            self.gamma = self._gamma_estimate
        
        
        self.mu = self.proj_mu(self.mu + coeff * (self.state - self.mu))

        self.params_history['gamma'].append(self.gamma.copy())
        self.params_history['mu'].append(self.mu.copy())

        
        self.sigma = self.proj_sigma(self.sigma + coeff * (alpha - self.tau_bar))
        self.params_history['sigma'].append(self.sigma)
