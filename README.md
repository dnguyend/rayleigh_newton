Author: Du Nguyen

This folder contains the code for the paper 
# Rayleigh Quotient Iteration and convergence analysis of feasibility perturbed higher-order constrained iterations #

The main algorithms are implemented in the library in folder [core](https://github.com/dnguyend/rayleigh_newton/tree/master/core). The examples workbooks are in folder [colab](https://github.com/dnguyend/rayleigh_newton/tree/master/colab).

The four workbooks in folder colab are related to section 6.1 and 6.5 in the paper
* [UZPairsEigenTensor.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/UZPairsEigenTensor.ipynb) shows the computations for UZ pairs, the first part of section 6.1
*  [BPairsHomogeneousSurface.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/BPairsHomogeneousSurface.ipynb) Real $B$ pairs with constraint a surface defined by $B$.
*  [BPairsUnitaryLinearEigenTensor.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/BPairsUnitaryLinearEigenTensor.ipynb) Complex pairs, computed using linear constraint then renormalized unitarily. Last part of section 6.1
*  [SimpleRQI_RChebyshev.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/SimpleRQI_RChebyshev.ipynb) For section 6.5. A simple example showing the Hessian of the retraction is needed in the Chebyshev term.

# Files stored in github may have hidden cells. Just unhide the cells.
The files contain the latest runs. If you prefer running the cell - open in colab - (need a google account) then just run. May see some warnings that the files are not created by google. Just ignore.
