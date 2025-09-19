import enum
import logging
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

pdist2 = cdist


class KernelType(enum.Enum):
    POL = 0
    RBF = 1


class Strategy(enum.Enum):
    RANDOM = 0
    KMEANS = 1


def _kmeans(data: np.ndarray, m: int, iterations: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=m, n_init=1, max_iter=iterations)
    kmeans.fit(data)
    return kmeans.cluster_centers_


def INys(
    kernel: KernelType, kernel_parameter: float, data: np.ndarray, m: int, s: Strategy
):
    n, _ = data.shape

    if s == Strategy.KMEANS:
        center = _kmeans(data, m, 5)
    elif s == Strategy.RANDOM:
        dex = np.random.permutation(n)
        center = data[dex[:m], :]
    else:
        raise AttributeError(f"Unknown strategy: '{s}'")

    if kernel == KernelType.POL:
        W = np.dot(center, center.T)
        E = np.dot(data, center.T)
        W = W**kernel_parameter
        E = E**kernel_parameter
    elif kernel == KernelType.RBF:
        # Laplacian kernel (L1)
        W = np.exp(-pdist2(center, center, "cityblock") / kernel_parameter)
        E = np.exp(-pdist2(data, center, "cityblock") / kernel_parameter)
    else:
        raise AttributeError(f"Unknown kernel type: '{kernel}'")

    # Eigen decomposition
    va, Ve = eigh(W)
    pidx = np.where(va > 1e-6)[0]
    inVa = np.diag(va[pidx] ** (-0.5))

    G = E @ (Ve[:, pidx] @ inVa)
    return G


def fuzzy_cluster_reg(
    src_pt: np.ndarray,
    tgt_pt: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 50,
    theta: float = 0.5,
    beta: float = 0.5,
    _lambda: float = 0.1,
    epsilon: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    Nc, D_source = src_pt.shape
    Np, D_target = tgt_pt.shape
    assert D_source == D_target
    D = D_source
    NpD = Np * D

    # Merge centroids between the both point sets first
    src_pt_center = np.mean(src_pt, axis=0)
    tgt_pt_center = np.mean(tgt_pt, axis=0)
    center_dist = tgt_pt_center - src_pt_center
    src_pt += center_dist

    # Initialize the variance sigma^2
    sigma2 = (
        Np * np.trace(src_pt @ src_pt.T)
        + Nc * np.trace(tgt_pt @ tgt_pt.T)
        - 2 * np.sum(src_pt, axis=0) @ np.sum(tgt_pt, axis=0).T
    ) / (D * Nc * Np)

    # Compute the Gramm matrix and low-rank decomposision by the improved Nystrom
    theta = 0.5
    # Number of landmark points, the larger the slower but more accurate
    m = int(np.ceil(0.3 * Nc))

    Q = INys(KernelType.RBF, theta, src_pt, m, Strategy.KMEANS)

    # Parameter initialization
    W = np.zeros((Nc, D))

    iter = 0
    ntol = tol + 10
    Loss = 1

    T = src_pt
    F = tgt_pt

    alpha = np.ones((1, Nc))
    onesUy = np.ones((Nc, 1))
    onesUx = np.ones((Np, 1))

    logging.debug(f"Initial {sigma2=:.8f}")

    while ntol > tol and iter < max_iter and sigma2 > 1e-8:
        Loss_old = Loss
        QtW = Q.T @ W

        fuzzy_dist = np.exp(-pdist2(F, T, "sqeuclidean") / (sigma2 * beta)) * alpha
        sum_fuzzy_dist = 1.0 / (np.sum(fuzzy_dist, axis=1))
        U = fuzzy_dist * sum_fuzzy_dist[:, np.newaxis] + epsilon

        U1 = U @ onesUy
        Ut1 = U.T @ onesUx
        dU = np.diag(U1.flatten())
        dUt = np.diag(Ut1.flatten())
        dUtQ = dUt @ Q
        Uttgt = U.T @ tgt_pt

        P = Uttgt - dUt @ src_pt
        A = _lambda * sigma2 * np.eye(Q.shape[1]) + Q.T @ dUtQ

        W = 1.0 / (_lambda * sigma2) * (P - dUtQ @ (np.linalg.inv(A) @ (Q.T @ P)))

        wdist_pt2center = np.abs(
            np.trace(F.T @ dU @ F + T.T @ dUt @ T - 2 * Uttgt.T @ T)
        )

        alpha = np.sum(U, axis=0) / Np + epsilon
        H_U = np.sum(U * np.log(U))
        H_alpha = Np * alpha @ np.log(alpha).T
        KL_U_alpha = H_U - H_alpha

        Loss = (
            (1.0 / sigma2) * wdist_pt2center
            + NpD * np.log(sigma2)
            + 0.5 * _lambda * np.trace(QtW.T @ QtW)
            + beta * KL_U_alpha
        )
        ntol = np.abs((Loss - Loss_old) / Loss)

        T = src_pt + Q @ (Q.T @ W)
        sigma2 = wdist_pt2center / NpD

        iter += 1
        logging.debug(f"{iter=:3d} {ntol=:.8f} {sigma2=:.8f}")
    return alpha, T
