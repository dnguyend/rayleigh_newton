import numpy as np


def tv_mode_product(T, x, modes) -> np.ndarray:
    """ Evaluating T subsituting x for the last modes times indices
    """
    v = T
    for i in range(modes):
        v = np.tensordot(v, x, axes=1)
    return v
    

def gen_random_anti_symmetric(k):
    p_raw = np.random.randint(1, 10, (k * (k-1)) // 2)
    p = np.zeros((k, k))
    start = 0
    for i in range(k-1):
        p[i, i+1:] = p_raw[start:start+k-i-1]
        p[i+1:, i] = -p_raw[start:start+k-i-1]
        start += k-i-1
    return p


def gen_random_symmetric(k):
    p_raw = np.random.randint(1, 10, (k * (k+1)) // 2)
    p = np.zeros((k, k))
    start = 0
    for i in range(k-1):
        p[i, i+1:] = p_raw[start:start+k-i-1]
        p[i+1:, i] = p_raw[start:start+k-i-1]
        start += k-i-1
    np.fill_diagonal(p, p_raw[-k:])
    return p


def gen_random_symmetric_pos(k):
    p_raw = np.random.randint(1, 10, (k * (k+1)) // 2)
    p = np.zeros((k, k))
    start = 0
    for i in range(k-1):
        p[i, i+1:] = p_raw[start:start+k-i-1]
        start += k-i-1
    np.fill_diagonal(p, p_raw[-k:])
    
    return np.dot(p.T, p)


def gen_random_real_eigen(k):
    V = np.random.randint(-10, 10, (k, k))
    D = np.diag(np.random.randint(-10, 10, k))
    return np.dot(np.dot(V, D), np.linalg.inv(V))
        

def transpose_tensor(shape):
    """ representing x.T
    """
    n, p = shape
    ret = np.zeros((n*p, n*p), dtype=float)
    for i in range(n):
        for j in range(p):
            ret[j*n+i, i*p+j] = 1

    return ret


def tensor_to_matrix_reshape(T, n, p):
    """M is of shape (n, p, n, p)
    acting by contracting the last 2
    we represent this as a matrix
    """
    return T.reshape(n*p, n*p)


def vech(mat):
    return mat.take(_triu_indices(len(mat)))


def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result


def triu_diag_vech_indices(p):
    rp = np.arange(p)
    return rp * p - (rp * (rp - 1)) // 2


def vech_project_tensor(p):
    """ (p*(p+1)2, p*p) matrix
    such that unvech(T @ m.reshape(-1)) = m if m is symmetric
    and zero if m is anti-symmetric
    """
    ret = np.zeros(((p*(p+1))//2, p*p), dtype=float)
    rows, cols = np.triu_indices(p)
    ret[np.arange(len(rows)), rows*p+cols] = .5
    ret[np.arange(len(rows)), cols*p+rows] = .5
    drows, dcols = np.diag_indices(p)
        
    ret[triu_diag_vech_indices(p), drows*p+dcols] = 1
    return ret


def vech_embedding_tensor(p):
    """ (p*p, p(p+1)/2) tensor
    tensor such that m = (T @ vech(m)).reshape(k, k)
    if m is symmetric
    """
    ret = np.zeros((p*p, (p*(p+1)) // 2), dtype=float)
    rows, cols = np.triu_indices(p)
    ret[rows*p+cols, range(len(rows))] = 1
    ret[cols*p+rows, range(len(rows))] = 1
    return ret    


def generate_polynomial_tensor(k, m):
    """Tensor representing a vector polynomial function
    """
    A = np.full(tuple(m*[k]), np.nan)
    for i in range(k):        
        A[i, None] = generate_symmetric_tensor(k, m-1)
    return A


def generate_symmetric_tensor(k, m):
    """Generating symmetric tensor size k,m
    """
    A = np.full(tuple(m*[k]), np.nan)
    current_idx = np.zeros(m, dtype=int)
    active_i = m - 1
    A[tuple(current_idx)] = np.random.rand()
    while True:
        if current_idx[active_i] < k - 1:
            current_idx[active_i] += 1
            if np.isnan(A[tuple(current_idx)]):
                i_s = tuple(sorted(current_idx))
                if np.isnan(A[i_s]):
                    A[i_s] = np.random.rand()
                    # print('Doing %s' % str(i_s))
                A[tuple(current_idx)] = A[i_s]
                # print('Doing %s' % str(current_idx))
        elif active_i == 0:
            break
        else:
            next_pos = np.where(current_idx[:active_i] < k-1)[0]
            if next_pos.shape[0] == 0:
                break
            current_idx[next_pos[-1]] += 1
            current_idx[next_pos[-1]+1:] = 0
                        
            active_i = m - 1
            if np.isnan(A[tuple(current_idx)]):
                i_s = tuple(sorted(current_idx))
                if np.isnan(A[i_s]):
                    A[i_s] = np.random.rand()
                    # print('Doing %s' % str(i_s))
                A[tuple(current_idx)] = A[i_s]
                # print('Doing %s' % str(current_idx))
    return A


def generate_symmetric_tensor_from_poly(X, expr):
    """Generating symmetric tensor size k,m from a vector of variables
    X: sympy matrix vector of size (n)
    expression. Homogenous polynomial in the variables, entries of X
    for example: X = sp.Matrix([x1, x2, x3])
    expr = x1**3 + x2**3 - x3**3
    return the tensor evaluated to that expression
    """
    import sympy as sp
    
    k = X.shape[0]

    expr_list = sp.Poly(expr, list(X)).coeffs()
    expr_monom = sp.Poly(expr, list(X)).monoms()
    
    m = sum(expr_monom[0])
    tot_terms = sum(X)**m

    divs = sp.Poly(tot_terms).coeffs()
    monom = sp.Poly(tot_terms).monoms()
    coeffs = len(monom)*[0]

    for jj in range(len(expr_monom)):
        idx = monom.index(expr_monom[jj])
        coeffs[idx] = expr_list[jj]

    def count_i_s(i_s):
        ret = k*[0]
        for i in i_s:
            ret[i] += 1
        return ret
    A = np.full(tuple(m*[k]), np.nan)
    current_idx = np.zeros(m, dtype=int)
    active_i = m - 1
    A[tuple(current_idx)] = coeffs[0]/divs[0]
    while True:
        if current_idx[active_i] < k - 1:
            current_idx[active_i] += 1
            if np.isnan(A[tuple(current_idx)]):
                i_s = tuple(sorted(current_idx))
                if np.isnan(A[i_s]):
                    cnt_i_s = tuple(count_i_s(i_s))
                    fidx = monom.index(cnt_i_s)
                    A[i_s] = coeffs[fidx] / divs[fidx]
                    # print('Doing %s' % str(i_s))
                A[tuple(current_idx)] = A[i_s]
                # print('Doing %s' % str(current_idx))
        elif active_i == 0:
            break
        else:
            next_pos = np.where(current_idx[:active_i] < k-1)[0]
            if next_pos.shape[0] == 0:
                break
            current_idx[next_pos[-1]] += 1
            current_idx[next_pos[-1]+1:] = 0
                        
            active_i = m - 1
            if np.isnan(A[tuple(current_idx)]):
                i_s = tuple(sorted(current_idx))
                if np.isnan(A[i_s]):
                    cnt_i_s = tuple(count_i_s(i_s))
                    fidx = monom.index(cnt_i_s)
                    A[i_s] = coeffs[fidx] / divs[fidx]
                    # A[i_s] = np.random.rand()
                    # print('Doing %s' % str(i_s))
                A[tuple(current_idx)] = A[i_s]
                # print('Doing %s' % str(current_idx))
    return A
