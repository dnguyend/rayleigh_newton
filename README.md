Author: Du Nguyen

This folder contains the code for the paper 
# Rayleigh Quotient Iteration, cubic convergence and second covariant derivative #

The main algorithms are implemented in the library in folder [core](https://github.com/dnguyend/rayleigh_newton/tree/master/core). The examples workbooks are in folder [colab](https://github.com/dnguyend/rayleigh_newton/tree/master/colab).

The four workbooks in folder colab are related to section 6 in the paper
* [UZPairsEigenTensor.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/UZPairsEigenTensor.ipynb) shows the computations for UZ pairs, the first part of section 6.1 for complex pairs.
* [JuliaTensorRQI.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/JuliaTensorRQI.ipynb) Julia implementation of section 6.1.1 Provide a high precision implementation of the real RQI and Rayleigh-Chebyshev. Confirming quadratic/cubic convergence.
* [JuliaRQIQuadracticOnSphere.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/JuliaRQIQuadracticOnSphere.ipynb) Julia implementation of section 6.3, for the case of eigenvalues with a constant term. Using the Julia library Arblib for higher precision.
*  [SimpleRQI_RChebyshev.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/SimpleRQI_RChebyshev.ipynb) For section 6.5. A simple example showing the Hessian of the retraction is needed in the Chebyshev term. A symbolic calculation.

* The folder extras contains a few more examples.
  *  [BPairsHomogeneousSurface.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/BPairsHomogeneousSurface.ipynb) Tensor eigenvalue problem with real $B$ pairs with constraint a surface defined by $B$.
  *  [BPairsUnitaryLinearEigenTensor.ipynb](https://github.com/dnguyend/rayleigh_newton/blob/master/colab/BPairsUnitaryLinearEigenTensor.ipynb) Complex pairs, computed using linear constraint then renormalized unitarily. Last part of section 6.1

# Files stored in github may have hidden cells. Just unhide the cells.
The files contain the latest runs. If you prefer running the cell - open in colab - (need a google account) then just run. May see some warnings that the files are not created by google. Just ignore.
