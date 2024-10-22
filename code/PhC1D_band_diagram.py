"""
    MIT Licence, © Alexey A. Shcherbakov, 2024
"""
import numpy as np
from scipy.linalg import toeplitz, eig
import matplotlib.pyplot as plt
'''
plot dispersion of 1D two-layer photonic crystal
'''
def plot_1D_PhC_dispersion(eps1 : float, eps2 : float, d1 : float, d2 : float, \
                            n : int, nb : int) -> None :
    '''
    eps1 : permittivity of the first layer 
    eps2 : permittivity of the second layer
    d1 : thickness of the first layer
    d2 : thickness of the second layer
    n : number of positive Fourier harmonics (total number of harmonics is 2n+1)
    nb : number of bands to calculate
    '''
#    eps1 = 1 # permittivity of the first layer
#    eps2 = 4 # permittivity of the second layer
    period = d1 + d2 # period in length units
#    k = 0.1 # Bloch wavevector

#    n = 5 # number of positive Fourier harmonics (total number is 2n+1)
    N = 2*n # number of positive Fourier harmonics for epsilon decomposition

    nrow = np.linspace(-n, n, 2*n+1)
    Nrow = np.linspace(-N, N, 2*N+1)
    Nrow[N] = 1.0

      # Fourier vector of the permittivity:
    fv_eps = (eps2 - eps1) * np.sin((np.pi*d1/period)*Nrow) / ((np.pi)*Nrow)
    fv_eps[N] = (eps1*d1 + eps2*d2) / period
      # epsilon Fourier matrix:
    FM = toeplitz(fv_eps[N:], fv_eps[N::-1])

#    nb = 4 # number of bands to calculate
    kv = np.arange(0, 0.5, 0.001, dtype=float) # Bloch k-vectors in the units of pi/Lambda
    bands = np.zeros((kv.size, nb), dtype=float)

    # loop over k-space points:
    for ik in range(0, kv.size): # kv.size
        A = np.diag((kv[ik] + nrow)**2)
        eigenval, eigenvect = eig(A, FM)
        eigenval = np.sqrt(eigenval)
        indices_sorted = np.argsort(np.real(eigenval))
        b = eigenval.flat[indices_sorted]
        bands[ik,:] = np.real(b[0:nb])

      # plot the PhC band diagram
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #fig, ax = plt.subplots()
    for ib in range(0, nb):
        plt.plot(kv, bands[:,ib], color='blue')

      # empty lattice diagram
    c = np.sqrt( period / (d1*eps1 + d2*eps2) )
    for ib in range(0, nb):
        plt.plot(kv, c*(kv+ib), linestyle='dashed', color='grey', )
        plt.plot(kv, c*(-kv+ib), linestyle='dashed', color='grey')

    plt.xlabel(r"$k_x\Lambda/2\pi$")
    plt.ylabel(r"$\Lambda/\lambda$")
    plt.xlim([0, 0.5])
    plt.ylim([0, 1.05*np.max(bands[:,-1])])
    plt.show()

if __name__ == "__main__":
    plot_1D_PhC_dispersion(eps1=1, eps2=2.5, d1=0.1, d2=0.2, n=5, nb=4)