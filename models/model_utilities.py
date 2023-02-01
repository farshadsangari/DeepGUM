from scipy.stats import multivariate_normal
import numpy as np


def gaussian_eval(y_preds, y_trues, cov):
    gm_result = np.zeros(y_preds.shape)
    for i in range(gm_result.shape[1]):
        var = multivariate_normal(mean=y_preds[0, i], cov=cov)
        gm_result[0, i] = var.pdf(y_trues[0, i])
    return gm_result


def e_step(prior, gamma, y_trues, y_preds, cov):
    gm_result = gaussian_eval(y_preds, y_trues, cov)
    numerator = prior * gm_result
    denumerator = prior * gm_result + (1 - prior) * gamma
    result = numerator / denumerator
    result = result.T
    return result


def m_step(prior, rn_s, y_trues, y_preds):
    # y_trues dim : numpy d*n
    # rn_s dim: n*1
    n = y_trues.shape[1]
    d = y_trues.shape[0]
    delta_s = y_trues - y_preds
    diag_rns = np.diag(rn_s.ravel())

    # Compute Covariance Matrix
    cov_matrix = np.matmul(np.matmul(delta_s, diag_rns), delta_s.T)

    # Compute prior
    prior = rn_s.mean()

    # Compute gamma
    C1 = (1 / (n * (1 - prior))) * (np.matmul(delta_s, (1 - rn_s)))
    C2 = (1 / (n * (1 - prior))) * (np.matmul((delta_s**2), (1 - rn_s)))
    gamma_inverse = np.product(2 * np.sqrt(3 * (C2 - (C1**2))))
    gamma = 1 / gamma_inverse
    return cov_matrix, prior, gamma
