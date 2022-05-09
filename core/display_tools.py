"""Helper functions to display results in workbooks
"""

import numpy as np
import pandas as pd


def display_one(comb, i):
    return (comb.x)[i, :], (comb.lbd)[i], (comb.is_real)[i]


def display_all_real(comb)  :
    return pd.DataFrame(
        index=np.where(comb.is_real)[0],
        data=np.concatenate([comb.lbd[np.where(comb.is_real)[0]].real[:, None], comb.x[np.where(comb.is_real)[0], :].real], axis=1),
        columns =['lbd'] + [str(i) for i in range(comb.x.shape[1])])


def display_all_complex(comb)  :
    return pd.DataFrame(
        index=np.where(comb.is_real==False)[0],
        data=np.concatenate([comb.lbd[np.where(comb.is_real==False)[0]].real[:, None], comb.x[np.where(comb.is_real==False)[0], :]], axis=1),
        columns =['lbd'] + [str(i) for i in range(comb.x.shape[1])])
    
