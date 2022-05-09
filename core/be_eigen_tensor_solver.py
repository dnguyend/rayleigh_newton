"""Bpairs with unitary constraint and m = d
"""

import numpy as np
from types import SimpleNamespace
from numpy import concatenate, zeros, power, sqrt

from numpy.linalg import solve, norm
from .utils import tv_mode_product


def schur_form_B_tensor_rayleigh_orthogonal_same_order(
        T, B, max_itr, delta, x_init=None):
    """Schur form rayleigh chebyshev unitary
    T and x are complex. Constraint is x^H x = 1
    lbd is real
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    d = len(B.shape)
    if m != d:
        print("T and B must have the same order m=%d != d=%d" % (m, d))
        return
    n = T.shape[0]
    R = 1
    converge = False

    # if not given as input, randomly initialize
    if x_init is None:
        x_init = np.random.randn(n)
        x_init = x_init/norm(x_init)

    # init lambda_(k) and x_(k)
    x_k = x_init.copy()
    T_x_m_2 = tv_mode_product(T, x_k, m-2)
    T_x_m_1 = T_x_m_2 @ x_k

    B_x_m_2 = tv_mode_product(B, x_k, d-2)
    B_x_m_1 = B_x_m_2 @ x_k

    lbd = (B_x_m_1.T @ T_x_m_1)/norm(B_x_m_1)**2
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        rhs = concatenate(
            [B_x_m_1.reshape(-1, 1), T_x_m_1.reshape(-1, 1)-lbd*B_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2-lbd*(d-1)*B_x_m_2
        lhs = solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum((x_k * lhs[:, 1])) /
            np.sum((x_k * lhs[:, 0]))) - lhs[:, 1]
        # print('%f %f %f' % (norm(lhs[:, 0]), norm(lhs[:, 1]), norm(np.sum(x_k.conjugate()*y))))
        x_k_n = (x_k + y) / norm(x_k + y)

        # x_k_n = (x_k + y)/(np.linalg.norm(x_k + y))

        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n

        T_x_m_2 = tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k
        B_x_m_2 = tv_mode_product(B, x_k, d-2)
        B_x_m_1 = B_x_m_2 @ x_k

        lbd = (B_x_m_1.T @ T_x_m_1)/norm(B_x_m_1)**2
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k
    err = norm(tv_mode_product(
        T, x, m-1) - lbd * tv_mode_product(
        B, x, d-1))
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge, err


def schur_form_B_tensor_rayleigh_unitary_same_order(
        T, B, max_itr, delta, x_init=None):
    """Schur form rayleigh chebyshev unitary
    T and x are complex. Constraint is x^H x = 1
    lbd is real
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    d = len(B.shape)
    if m != d:
        print("T and B must have the same order m=%d != d=%d" % (m, d))
        return
    n = T.shape[0]
    R = 1
    converge = False

    # if not given as input, randomly initialize
    if x_init is None:
        x_init = np.random.randn(n) + 1j*np.random.randn(n)
        x_init = x_init/norm(x_init)

    # init lambda_(k) and x_(k)
    x_k = x_init.copy()
    T_x_m_2 = tv_mode_product(T, x_k, m-2)
    T_x_m_1 = T_x_m_2 @ x_k

    B_x_m_2 = tv_mode_product(B, x_k, d-2)
    B_x_m_1 = B_x_m_2 @ x_k

    lbd = (B_x_m_1.conjugate().T @ T_x_m_1)/norm(B_x_m_1)**2
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        rhs = concatenate(
            [B_x_m_1.reshape(-1, 1), T_x_m_1.reshape(-1, 1)-lbd*B_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2-lbd*(d-1)*B_x_m_2
        lhs = solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum((x_k.conjugate() * lhs[:, 1])) /
            np.sum((x_k.conjugate() * lhs[:, 0]))) - lhs[:, 1]
        # print('%f %f %f' % (norm(lhs[:, 0]), norm(lhs[:, 1]), norm(np.sum(x_k.conjugate()*y))))
        x_k_n = (x_k + y) / norm(x_k + y)

        # x_k_n = (x_k + y)/(np.linalg.norm(x_k + y))

        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n

        T_x_m_2 = tv_mode_product(T, x_k, m-2)
        T_x_m_1 = T_x_m_2 @ x_k
        B_x_m_2 = tv_mode_product(B, x_k, d-2)
        B_x_m_1 = B_x_m_2 @ x_k

        lbd = (B_x_m_1.conjugate().T @ T_x_m_1)/norm(B_x_m_1)**2
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k
    err = norm(tv_mode_product(
        T, x, m-1) - lbd * tv_mode_product(B, x, d-1))
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge, err


def complex_eigen_cnt(m, d, n):
    if m == d:
        return n*power(m-1, n-1)
    return (power(m-1, n)-power(d-1, n)) // (m-d)


def find_eig_cnt(all_eig):
    first_nan = np.where(np.isnan(all_eig.x))[0]
    if first_nan.shape[0] == 0:
        return None
    else:
        return first_nan[0]

    
def normalize_real(lbd, x, m, d, tol):
    """ First try to make it to a real pair
    if not possible. If not then make lambda real
    return is_self_conj, is_real, new_lbd, new_x
    """
    u = (sqrt(x @ x).conjugate())
    new_x = x * u
    # if np.sum(np.abs(new_x.imag)) < tol:
    if np.abs(np.abs(u) - 1) < tol:
        # try to flip. if u **(m-d) > 0 use it:
        # lbd_factor = lbd_factor.real
        new_x = (new_x.real + 0.j)/norm(new_x.real)
        return True, lbd, new_x

    return False, lbd, x


def insert_eigen(all_eig, x, lbd, eig_cnt, m, d, tol, disc):
    """
    force eigen values to be positive if possible
    if x is not similar to a vector in all_eig.x
    then:
       insert pair x, conj(x) if x is not self conjugate
       otherwise insert x
    all_eig has a structure: lbd, x, is_self_conj, is_real
    """
    is_real, norm_lbd, norm_x = normalize_real(
        lbd, x, m, d, tol)

    if is_real:
        good_x = [norm_x]
        good_lbd = [norm_lbd]
    else:
        good_x = [norm_x, norm_x.conjugate()]
        good_lbd = [norm_lbd, norm_lbd.conjugate()]
    nct = 0
    for xx in good_x:
        #  factors = all_eig.x[:eig_cnt+nct, :] @ xx.conjugate()
        # fidx = np.where(np.abs(factors ** (m-2) - 1) < disc)[0]
        if eig_cnt+nct >= all_eig.lbd.shape[0]:
            print("more values found than expected. Likely having a degenerate zero")
            break
        
        factors = all_eig.x[:eig_cnt+nct, :] @ xx.conjugate()
        fidx = np.where(np.abs(np.abs(factors)-1) < disc)[0]
        if fidx.shape[0] == 0:
            all_eig.lbd[eig_cnt+nct] = norm_lbd
            all_eig.x[eig_cnt+nct] = xx
            all_eig.is_self_conj[eig_cnt+nct] = False
            all_eig.is_real[eig_cnt+nct] = is_real
            nct += 1

    return eig_cnt + nct


def find_all_unitary_eigenpair_same_order(
        all_eig, eig_cnt, A, B, max_itr, max_test, tol, disc):
    """ output is the table of results
     2n*+2 columns: lbd, is self conjugate, x_real, x_imag
    This is the raw version, since the output vector x
    is not yet normalized to be real when possible
    """
    n = A.shape[0]
    m = len(A.shape)
    d = len(B.shape)
    if m != d:
        print("cannot deal with m=%d != n=%d" % (m, d))
        return None, None

    n_eig = complex_eigen_cnt(m, d, n)
    if all_eig is None:
        all_eig = SimpleNamespace(
            lbd=np.full((n_eig), np.nan, dtype=float),
            x=np.full((n_eig, n), np.nan, dtype=complex),
            is_self_conj=zeros((n_eig), dtype=bool),
            is_real=zeros((n_eig), dtype=bool))
        eig_cnt = 0
    elif eig_cnt is None:
        eig_cnt = find_eig_cnt(all_eig)
        if eig_cnt is None:
            return all_eig
    if all_eig is None:
        all_eig = SimpleNamespace(
            lbd=np.full((n_eig), np.nan, dtype=complex),
            x=np.full((n_eig, n), np.nan, dtype=complex),
            is_self_conj=zeros((n_eig), dtype=bool),
            is_real=zeros((n_eig), dtype=bool))
        eig_cnt = 0
    elif eig_cnt is None:
        eig_cnt = find_eig_cnt(all_eig)
        if eig_cnt is None:
            return all_eig

    for jj in range(max_test):
        x0r = np.random.randn(2*n-1)
        # x0r = np.random.randint(-10, 10, 2*n-1)*1.
        x0r /= norm(x0r)
        x0 = x0r[:n] + 0.j
        x0[1:] = x0[1:] + x0r[n:] * 1.j
        # x0[-1] = np.abs(x0[-1])
        # if there are odd numbers left,
        # try to find a real root
        draw = np.random.uniform(0, 1, 1)
        # 50% try real root
        if True and (draw < .5) and ((n_eig - eig_cnt) % 2 == 1):
            try:
                x_r, lbd, ctr, converge, err = schur_form_B_tensor_rayleigh_unitary_same_order(
                    A, B, max_itr, tol, x_init=x0.real)
                x = x_r + 1j * zeros((x_r.shape[0]))
            except Exception as e:
                print(e)
                continue
        else:
            try:
                x, lbd, ctr, converge, err =\
                    schur_form_B_tensor_rayleigh_unitary_same_order(
                        A, B, max_itr, tol, x_init=x0)
                # print(x0, x, lbd, ctr, converge, err)
            except Exception as e:
                print(e)
                continue
        old_eig = eig_cnt        
        if (err < tol):
            eig_cnt = insert_eigen(all_eig, x, lbd, eig_cnt, m, d, tol, disc)
        if eig_cnt == n_eig:
            break
        # elif (eig_cnt > old_eig) and (eig_cnt % 10 == 0):
        elif (eig_cnt > old_eig) and True:
            print('Found %d eigenpairs' % eig_cnt)
    return SimpleNamespace(
            lbd=all_eig.lbd[:eig_cnt],
            x=all_eig.x[:eig_cnt, :],
            is_self_conj=all_eig.is_self_conj[:eig_cnt],
            is_real=all_eig.is_real[:eig_cnt]), eig_cnt

