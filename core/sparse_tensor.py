"""Module implementing
"""
import numpy as np
from numpy import concatenate, eye
from numpy.linalg import norm, solve


class SparseTensor():
    """
    In this version of the code, the tensor is not given by a dense full tensor
    but from a function parametrized by a tensor t_mat of size n times (....)
    function is evaluated from t_mat.
    t_mat is typically much smaller than the full tensor of order m.
    This allows us to deal with tensors of higher dimension
    The method provide an iteration to give one eigenvalue
    
    To use, derive the method, supplying eval, hess_eval    

    In tensor form
    eval()[0] is the value of the function, tv_mode_product(m)

    eval()[1] is the gradient, tv_mode_product(m-1) = 1/m*eval()[1]
    hess_eval is the hessian, tv_mode_product(m-2) = 1/(m*(m-1))*hess_eval()
    """
    def __init__(self, t_mat, m):
        self.n = t_mat.shape[0]
        self.m = m
    
    def rand(self):
        """ A random vector of size n
        """
        return np.random.randn(self.n)

    def eval(self, x):
        """ evaluating the multilinear function and its gradient at x
        """
        raise NotImplementedError

    def hess_eval(self, x):
        """ Hessian of the multilinear funciton at x
        """
        raise NotImplementedError
    
    def schur_form_rayleigh(
            self, max_itr, delta, x_init=None, save_err=False):
        """Schur form rayleigh
        the tensor is given by the multilinear function class tns,
        which can evaluate the tensor, gradient and hessian
        """
        # get tensor dimensionality and order
        n, m = self.n, self.m

        err = 1
        converge = False

        # if not given as input, randomly initialize
        if x_init is None:
            x_init = np.random.randn(n)
            x_init = x_init/norm(x_init)

        # init lambda_(k) and x_(k)
        lbd, grad = self.eval(x_init)
        x_k = x_init.copy()
        ctr = 0
        if save_err:
            err_save = []

        while (err > delta) and (ctr < max_itr):
            # compute T(I,I,x_k,...,x_k), T(I,x_k,...,x_k) and g(x_k)
            t_x_m_2 = 1/(m*(m-1))*self.hess_eval(x_k)
            # T_x_m_1 = T_x_m_2 @ x_k
            t_x_m_1 = 1/m*grad
            rhs = concatenate(
                [x_k.reshape(-1, 1), t_x_m_1.reshape(-1, 1)], axis=1)

            # compute Hessian H(x_k)
            hess = (m-1)*t_x_m_2-lbd*eye(n)
            lhs = solve(hess, rhs)

            # fix eigenvector
            y = lhs[:, 0] * (
                np.sum(x_k * lhs[:, 1]) / np.sum(x_k * lhs[:, 0])) - lhs[:, 1]
            x_k_n = (x_k + y)/(norm(x_k + y))

            #  update residual and lbd
            err = norm(x_k-x_k_n)
            x_k = x_k_n
            lbd, grad = self.eval(x_k)
            # print('ctr=%d lbd=%f err=%.2e' % (ctr, lbd, err))
            if save_err:
                err_save.append((lbd, err))
            ctr += 1
        x = x_k
        if ctr < max_itr:
            converge = True
        if save_err:
            return x, lbd, ctr, converge, np.array(err_save)
        return x, lbd, ctr, converge


class ThreeFactorSparseTensor(SparseTensor):
    """Tensor T of size n times 2
    function is evaluated as 
    sum t_[i,0] x_i^m + sum t_[i,1]x_i^(m-2)*x_[i-1]*x_[i-2]

    In tensor form
    eval()[0] is the value of the function, tv_mode_product(m)

    eval()[1] is the gradient, tv_mode_product(m-1) = 1/m*eval()[1]
    hess_eval is the hessian, tv_mode_product(m-2) = 1/(m*(m-1))*hess_eval()
    """
    def __init__(self, t_mat, m):
        self.t_mat = t_mat
        self.m = m
        self.n = t_mat.shape[0]

    def rand(self):
        """ A random vector of size n
        """
        return np.random.randn(self.n)

    def eval(self, x):
        """ evaluating the tensor at x
        """
        t_mat, m = self.t_mat, self.m
        ret = np.sum(x**m*t_mat[:, 0])
        ret1 = t_mat[:-2, 1]*x[:-2]**(m-2)*x[1:-1]*x[2:]
        grad = m*x**(m-1)*t_mat[:, 0]
        if m > 3:
            grad[:-2] += (m-2)*t_mat[:-2, 1]*x[:-2]**(m-3)*x[1:-1]*x[2:]
            grad[1:-1] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[2:]
            grad[2:] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[1:-1]
        else:
            grad[:-2] += (m-2)*t_mat[:-2, 1]*x[1:-1]*x[2:]
            grad[1:-1] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[2:]
            grad[2:] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[1:-1]
        return ret + np.sum(ret1), grad

    def hess_eval(self, x):
        """ Hessian of the tensor at x
        """
        t_mat, m, n = self.t_mat, self.m, self.n

        hess = np.diag(m*(m-1)*x**(m-2)*t_mat[:, 0])
        # grad[:-2] += (m-2)*t_mat[:-2, 1]*x[:-2]**(m-3)*x[1:-1]*x[2:]
        #  grad[1:-1] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[2:]
        #  grad[2:] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[1:-1]
        if m > 3:
            hess[np.arange(n-2), np.arange(n-2)] +=\
                (m-2)*(m-3)*t_mat[:-2, 1]*x[:-2]**(m-4)*x[1:-1]*x[2:]

            hess[np.arange(n-2), np.arange(1, n-1)] += (m-2)*t_mat[:-2, 1]*x[:-2]**(m-3)*x[2:]
            hess[np.arange(n-2), np.arange(2, n)] += (m-2)*t_mat[:-2, 1]*x[:-2]**(m-3)*x[1:-1]

            hess[np.arange(1, n-1), np.arange(n-2)] += (m-2)*t_mat[:-2, 1]*x[:-2]**(m-3)*x[2:]
            # hess[np.arange(1, n-1), np.arange(1, n-1)] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[2:]
            hess[np.arange(1, n-1), np.arange(2, n)] += t_mat[:-2, 1]*x[:-2]**(m-2)

            hess[np.arange(2, n), np.arange(n-2)] += (m-2)*t_mat[:-2, 1]*x[:-2]**(m-3)*x[1:-1]
            hess[np.arange(2, n), np.arange(1, n-1)] += t_mat[:-2, 1]*x[:-2]**(m-2)
            # hess[2:] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[1:-1]
        else:
            # hess[np.arange(n-2), np.arange(n-2)] += (m-2)*(m-3)*t_mat[:-2, 1]*x[:-2]**(m-4)*x[1:-1]*x[2:]
            hess[np.arange(n-2), np.arange(1, n-1)] += (m-2)*t_mat[:-2, 1]*x[:-2]**(m-3)*x[2:]
            hess[np.arange(n-2), np.arange(2, n)] += (m-2)*t_mat[:-2, 1]*x[:-2]**(m-3)*x[1:-1]

            hess[np.arange(1, n-1), np.arange(n-2)] += (m-2)*t_mat[:-2, 1]*x[2:]
            # hess[np.arange(1, n-1), np.arange(1, n-1)] += t_mat[:-2, 1]*x[:-2]**(m-2)*x[2:]
            hess[np.arange(1, n-1), np.arange(2, n)] += t_mat[:-2, 1]*x[:-2]**(m-2)

            hess[np.arange(2, n), np.arange(n-2)] += (m-2)*t_mat[:-2, 1]*x[1:-1]
            hess[np.arange(2, n), np.arange(1, n-1)] += t_mat[:-2, 1]*x[:-2]**(m-2)

        return hess
