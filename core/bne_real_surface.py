"""Eigen Bpairs where constraint is described by an eigen surface
Returning individual pairs only
"""

import numpy as np
from numpy import concatenate, tensordot

from numpy.linalg import solve, norm
from .utils import tv_mode_product


def rroots(v, d):
    """ real root of v. Works when v < 0 and d is odd
    """
    if (v < 0) and (d % 2 == 0):
        return np.nan
    return np.sign(v)*np.abs(v)**(1/d)


class Bfeasible(object):
    """ Class describing a real constraint surface
    defined by a homogenous equation
    """
    def __init__(self, B):
        self.B = B    
        self.n = B.shape[0]
        self.d = len(B.shape)

    def rand(self):
        while True:
            x = np.random.randn(self.n)
            vv = tv_mode_product(self.B, x, modes=self.d)
            if vv > 0:
                break
        return x / rroots(vv, self.d)

    def proj_tan(self, x, omg):
        V = tv_mode_product(self.B, x, modes=self.d-1)
        return omg - x*np.sum(V*omg)

    def Pi(self, x, omg):
        V = tv_mode_product(self.B, x, modes=self.d-1)
        return omg - V*np.sum(x*omg)

    def randvec(self, x):
        return self.proj_tan(x, np.random.randn(self.n))

    def tensor_rtr(self, x, eta):
        v = x+eta
        V = tv_mode_product(self.B, v, modes=self.d)
        return v / rroots(V, self.d)

    def D_tensor_rtr(self, x, xi, eta):
        v = x+xi    
        VB1 = tv_mode_product(self.B, v, modes=self.d-1)
        VB = np.sum(VB1*v)
        return eta / rroots(VB, self.d) - v/VB/rroots(VB, self.d)*np.sum(VB1*eta)

    def H_tensor_rtr(self, x, eta):
        # Hessian of the retraction
        V2 = tv_mode_product(self.B, x, modes=self.d-2)
        return -(self.d-1)*x*np.sum(eta*(V2@eta))    


def schur_form_B_tensor_rayleigh_chebyshev(
        T, Bf, max_itr, delta, x_init=None, do_chebyshev=True):
    """Schur form rayleigh chebyshev 
    T and x are complex. Constraint is B(x^{[d]}) = 1
    """
    # get tensor dimensionality and order
    n_vec = T.shape
    m = len(n_vec)
    n = T.shape[0]
    B = Bf.B
    d = Bf.d
    R = 1    
    converge = False

    # if not given as input, randomly initialize
    if x_init is None:
        x_init = Bf.rand()        

    # init lambda_(k) and x_(k)
    x_k = x_init.copy()
    if do_chebyshev:
        T_x_m_3 = tv_mode_product(T, x_k, m-3)
        T_x_m_2 = tensordot(T_x_m_3, x_k, axes=1)
        B_x_m_3 =  tv_mode_product(Bf.B, x_k, d-3)
        B_x_m_2 =  tensordot(B_x_m_3, x_k, axes=1)
    else:
        T_x_m_2 = tv_mode_product(T, x_k, m-2)
        B_x_m_2 = tv_mode_product(Bf.B, x_k, d-2)

    T_x_m_1 = T_x_m_2 @ x_k
    B_x_m_1 = B_x_m_2 @ x_k

    lbd = np.sum(x_k.T * T_x_m_1)
    ctr = 0

    while (R > delta) and (ctr < max_itr):
        # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
        rhs = concatenate(
            [B_x_m_1.reshape(-1, 1), T_x_m_1.reshape(-1, 1) -lbd*B_x_m_1.reshape(-1, 1)], axis=1)

        # compute Hessian H(x_k)
        H = (m-1)*T_x_m_2 - (d-1)*B_x_m_2*lbd
        lhs = solve(H, rhs)

        # fix eigenvector
        y = lhs[:, 0] * (
            np.sum((B_x_m_1 * lhs[:, 1])) /
            np.sum((B_x_m_1 * lhs[:, 0]))) - lhs[:, 1]
        if do_chebyshev and (np.linalg.norm(y) < 30e-2):
            D_R_eta = m * np.sum(T_x_m_1@y)
            L_x_lbd = -(d-1)*(B_x_m_2@y) * D_R_eta            
            L_x_x = (m-1) * (m-2) * np.tensordot(T_x_m_3, y, axes=1) @ y -\
                (d-1)*(d-2)*np.tensordot(B_x_m_3, y, axes=1) @ y

            L_x_H_rtr = ((m-1)*T_x_m_2 - (d-1)*B_x_m_2*lbd)@Bf.H_tensor_rtr(x_k, y)
            # we only need L_x_x but put the other ones for completeness
            T_a = np.linalg.solve(H, 2*L_x_lbd + L_x_x + L_x_H_rtr)
            # T_a = np.linalg.solve(H, L_x_x)
            T_adj = T_a - lhs[:, 0] *\
                np.sum(B_x_m_1 * T_a) / np.sum(B_x_m_1 * lhs[:, 0])
            # print(np.sum(B_x_m_1*T_adj))
            x_k_n = Bf.tensor_rtr(x_k, y - 0.5*T_adj)
        else:
            x_k_n = Bf.tensor_rtr(x_k, y)

        #  update residual and lbd
        R = norm(x_k-x_k_n)
        x_k = x_k_n
        # import pdb
        # pdb.set_trace()
        if do_chebyshev:
            T_x_m_3 = tv_mode_product(T, x_k, m-3)
            T_x_m_2 = np.tensordot(T_x_m_3, x_k, axes=1)
            
            B_x_m_3 = tv_mode_product(Bf.B, x_k, d-3)
            B_x_m_2 = np.tensordot(B_x_m_3, x_k, axes=1)
            B_x_m_1 = B_x_m_2 @ x_k
        else:
            T_x_m_2 = tv_mode_product(T, x_k, m-2)
            B_x_m_2 = tv_mode_product(Bf.B, x_k, d-2)
        T_x_m_1 = T_x_m_2 @ x_k            
        B_x_m_1 = B_x_m_2 @ x_k

        lbd = np.sum(x_k * T_x_m_1)
        # print('ctr=%d lbd=%f' % (ctr, lbd))
        ctr += 1
    x = x_k
    err = norm(tv_mode_product(
        T, x, m-1) - lbd * tv_mode_product(
        Bf.B, x, d-1))
    if ctr < max_itr:
        converge = True

    return x, lbd, ctr, converge, err


def Rayp(T, Bf, x, eta):
    # FF = F(x)
    m = len(T.shape)
    VA2 = tv_mode_product(T, x, m-2)
    VA1 = VA2@x
    return np.sum(eta*VA1) + (m-1)*np.sum(x*(VA2@eta))
    
    # return np.sum(eta*FF[0]) + np.sum(x*(FF[2]@eta+FF[5]@eta*FF[1]))

    
def GL(T, Bf, x, eta):
    B = Bf.B
    m = len(T.shape)
    d = len(B.shape)
    VT = eval(T, x)
    VB = eval(B, x)
    lbd = np.sum(x*VT[1])        
    rp = np.sum(eta*VT[1]) + (m-1)*np.sum(x*(VT[2]@eta))
    # print(rp)

    if m == 2:
        G = - (d-1)*(d-2)*lbd*np.tensordot(np.tensordot(VB[3], eta, axes=1), eta, axes=1) \
            + 2*(d-1)*VB[2]@eta*rp \
            + ((m-1)*VT[2] - (d-1)*VB[2]*lbd)@Bf.H_tensor_rtr(x, eta)
        
        return G
    G = (m-1)*(m-2)*np.tensordot(np.tensordot(VT[3], eta, axes=1), eta, axes=1) \
        - (d-1)*(d-2)*lbd*np.tensordot(np.tensordot(VB[3], eta, axes=1), eta, axes=1) \
        + 2*(d-1)*VB[2]@eta*rp \
        + ((m-1)*VT[2] - (d-1)*VB[2]*lbd)@Bf.H_tensor_rtr(x, eta)
    return G


def Lx(T, Bf, x, lbd):
    n_vec = T.shape
    m = len(n_vec)
    # n = T.shape[0]
    d = Bf.d
    VT2 = tv_mode_product(T, x, m-2)
    VB2 = tv_mode_product(Bf.B, x, d-2)
    return (m-1)*VT2 - (d-1)*VB2*lbd


def Lxx(T, Bf, x, lbd):
    n_vec = T.shape
    m = len(n_vec)
    # n = T.shape[0]
    d = Bf.d
    VT3 = tv_mode_product(T, x, m-3)
    VB3 = tv_mode_product(Bf.B, x, d-3)  
    return (m-2)*(m-1)*VT3 - (d-2)*(d-1)*VB3*lbd
