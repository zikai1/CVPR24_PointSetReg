import logging
from typing import Tuple

import cupy as cp
import numpy as np

from .fuzzyclusterreg import INys, KernelType, Strategy, pdist2


def fuzzy_cluster_reg_gpu(
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

    # Compute the Gramm matrix and low-rank decomposision by the improved Nystrom
    theta = 0.5
    # Number of landmark points, the larger the slower but more accurate
    m = int(np.ceil(0.3 * Nc))

    Q = cp.array(INys(KernelType.RBF, theta, src_pt, m, Strategy.KMEANS))
    ident_Q = cp.eye(Q.shape[1])

    src_pt = cp.array(src_pt)
    tgt_pt = cp.array(tgt_pt)

    # Initialize the variance sigma^2
    sigma2 = (
        Np * cp.trace(src_pt @ src_pt.T)
        + Nc * cp.trace(tgt_pt @ tgt_pt.T)
        - 2 * cp.sum(src_pt, axis=0) @ cp.sum(tgt_pt, axis=0).T
    ) / (D * Nc * Np)

    # Parameter initialization
    W = cp.zeros((Nc, D))

    iter = 0
    ntol = tol + 10
    Loss = 1

    T = cp.asarray(src_pt)
    F = cp.asarray(tgt_pt)

    alpha = cp.ones((1, Nc))

    logging.debug(f"Initial {sigma2=:.8f}")

    while ntol > tol and iter < max_iter and sigma2 > 1e-8:
        Loss_old = Loss
        QtW = Q.T @ W

        fuzzy_dist = (
            cp.exp(-cp.array(pdist2(F.get(), T.get(), "sqeuclidean")) / (sigma2 * beta))
            * alpha
        )
        sum_fuzzy_dist = cp.sum(fuzzy_dist, axis=1, keepdims=True)
        U = fuzzy_dist / sum_fuzzy_dist + epsilon

        U1 = cp.sum(U, axis=1)
        Ut1 = cp.sum(U.T, axis=1)
        dU = cp.diag(U1.flatten())
        dUt = cp.diag(Ut1.flatten())
        dUtQ = dUt @ Q
        Uttgt = U.T @ tgt_pt

        P = Uttgt - dUt @ src_pt
        A = _lambda * sigma2 * ident_Q + Q.T @ dUtQ
        W = (P - dUtQ @ (cp.linalg.inv(A) @ (Q.T @ P))) / (_lambda * sigma2)

        wdist_pt2center = cp.abs(
            cp.trace(F.T @ dU @ F + T.T @ dUt @ T - 2 * Uttgt.T @ T)
        )

        alpha = cp.sum(U, axis=0) / Np + epsilon
        H_U = cp.sum(U * cp.log(U))
        H_alpha = Np * alpha @ cp.log(alpha).T
        KL_U_alpha = H_U - H_alpha

        Loss = (
            wdist_pt2center / sigma2
            + NpD * cp.log(sigma2)
            + 0.5 * _lambda * cp.trace(QtW.T @ QtW)
            + beta * KL_U_alpha
        )
        ntol = cp.abs((Loss - Loss_old) / Loss)

        T = src_pt + Q @ (Q.T @ W)
        sigma2 = wdist_pt2center / NpD

        iter += 1
        logging.debug(f"{iter=:3d} {ntol=:.8f} {sigma2=:.8f}")
    return alpha, cp.asnumpy(T)
