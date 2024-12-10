"""
    MIT Licence, © Alexey A. Shcherbakov, 2024
"""
import numpy as np
from scipy.linalg import toeplitz, eig
from scipy.special import jv
import matplotlib.pyplot as plt

  # calculate a 2D Toeplitz matrix from a 2D Fourier transform vector
def toeplitz2(M, nx, ny):
    CT = np.zeros((2*nx-1, ny, ny), dtype=complex)
    for i in range(0, 2*nx-1):
        CT[i,:,:] = toeplitz(M[i,ny-1:],M[i,ny-1::-1])
    T = np.zeros((nx*ny, nx*ny), dtype=complex)
    for i in range(0, nx):
        for j in range(0, nx):
            T[i*ny:(i+1)*ny,j*ny:(j+1)*ny] = CT[nx-1+i-j,:,:]
    return T

class PhotonicCrystal2DCyl:
    def __init__(self, lattice_type : str, radius : float, period : float, eps_cyl : float, eps_med : float) -> None:
        '''
        lattice_type: 'square' or 'hexagonal'
        radius: radius of cylindrical rods or voids comrising the photonic crystal
        period: crystal period in length units
        eps_cyl: permittivity of cylindrical rods or voids, float or complex
        eps_med: permittivity of surrounding medium, float or complex
        '''
        self.lattice_type = lattice_type
        self.radius = radius
        self.period = period
        self.eps_cyl = eps_cyl
        self.eps_med = eps_med

          # calculate the reciprocal lattice vectors and crystal filling factors
        if lattice_type == "square":
            self.fill_fact = np.pi * radius**2 / period**2; # volume fraction occupied by the cylinders
            self.K1 = np.array([1.0, 0.0])
            self.K2 = np.array([0.0, 1.0])
        elif lattice_type == "hexagonal":
            self.fill_fact = (2*np.pi/np.sqrt(3.0)) * radius**2 / period**2 # volume fraction occupied by the cylinders
            self.K1 = np.array([0.5, -0.5*np.sqrt(3.0)])
            self.K2 = np.array([0.5, 0.5*np.sqrt(3.0)])
        pass

    def calc_reciprocal_lattice(self, n_harm, k_bloch):
        '''
        calculate vector projections for the reciprocal lattice points
        n_harm: maximum number of Fourier harmonics in one dimension
        k_bloch: numpy array or list of two floats defining the Bloch wavevector
        '''
        [k1, k2] = np.meshgrid(np.linspace(0, n_harm-1, n_harm) - int(n_harm/2), \
                                np.linspace(0, n_harm-1, n_harm) - int(n_harm/2))
        self.Kx = k_bloch[0] + self.K1[0]*k1 + self.K2[0]*k2
        self.Ky = k_bloch[1] + self.K1[1]*k1 + self.K2[1]*k2
        self.Km = np.sqrt(self.Kx**2 + self.Ky**2)
        self.Kx = self.Kx.reshape(-1)
        self.Ky = self.Ky.reshape(-1)
        self.Km = self.Km.reshape(-1)
        pass

    def calc_fourier(self, n_harm):
        '''
        calculate the Fourier amplitude vector of the inverse dielectric permittivitty periodic coordinate function
        h_harm: total number of Fourier harmonics to calculate in one dimension
        '''
        rp = self.radius / self.period
        if self.lattice_type == 'square':
            C1 = 2.0*np.pi
            C2 = 0.0
        elif self.lattice_type == 'hexagonal':
            C1 = 4.0*np.pi/np.sqrt(3.0)
            C2 = 0.5

        FM = np.zeros([2*n_harm-1, 2*n_harm-1], dtype=float)

        I1, I2 = np.meshgrid(np.linspace(-n_harm+1, n_harm-1, 2*n_harm-1), np.linspace(-n_harm+1, n_harm-1, 2*n_harm-1))	
        tF = C1*rp*np.sqrt(I1**2 + I2**2 - 2.0*C2*I1*I2)
        ind = np.abs(tF) > 1.0e-8
        tF[ind] = C1*(rp**2)*jv(1, tF[ind]) / tF[ind]
        tF[~ind] = 0.5*C1*(rp**2)
        FM = (1.0/self.eps_cyl - 1.0/self.eps_med) * tF
        FM[n_harm-1, n_harm-1] += 1.0/self.eps_med

        return FM

    def get_eig_mat(self, n_harm, polarization, TM):
        '''
        fill the matrix for the 2D photonic crystal eigenvalue problem
        h_harm: total number of Fourier harmonics to calculate in one dimension
        polarization: 'TE', 's' or 'TM', 'p'
        TM: Toeplitz matrix of the inverse dielectric permittivitty
        '''
        if polarization == 'TE' or polarization == 's':
            M = TM * (self.Kx.reshape([n_harm*n_harm,1]) * self.Kx.reshape([1,n_harm*n_harm]) \
                      + self.Ky.reshape([n_harm*n_harm,1]) * self.Ky.reshape([1,n_harm*n_harm]))
        elif polarization == 'TM' or polarization == 'p':
            M = TM * self.Km.reshape([n_harm*n_harm,1]) * self.Km.reshape([1,n_harm*n_harm])
        return M

#########################################################################################

def plot_circular_PhC_dispersion_2D():
      # parameters:
    radius = 0.18 # radius of cylinders
    period = 1 # period (P > 2R)
    eps_cyl = 11.56 # permittivity of cylinders
    eps_med = 1 # permittivity of the surrounding medium
    lattice_type = 'square' # either "square" or "hexagonal"

    polarization = 'TM' # % polarization
    n_harm = 21 # number of Fourier harmonics in one dimension

    n_band = 6 # set the number of bands to calculate:

      # initialization of the band diagram calculation:
    npt = 15 # number of points in each segment of the dispersion diagram to plot
    ntot = 3*npt+1 # total number of points
      # symmetry points, which define a trajectory in the k-space:
    if lattice_type == 'square':
        # psym = [[0.5, 0.5], [0.0, 0.0], [0.5, 0.0], [0.5, 0.5]] # M -> Gamma -> X -> M
        # symm_points = ['M','$\Gamma$','X','M']
        psym = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]] # Gamma -> X -> M -> Gamma
        symm_points = ['$\Gamma$','X','M','$\Gamma$']
    elif lattice_type == 'hexagonal':
        # psym = [[0.5, 0], [0.0, 0.0], [0.5, 0.5/np.sqrt(3.0)], [0.5, 0.0]] # M -> Gamma -> K -> M
        # symm_points = ['M','$\Gamma$','K','M']
        psym = [[0.0, 0.0], [0.5, 0], [0.5, 0.5/np.sqrt(3.0)], [0.0, 0.0]] # Gamma -> M -> K ->Gamma
        symm_points = ['$\Gamma$','M','K', '$\Gamma$']

      # initialize the crystal class
    crystal = PhotonicCrystal2DCyl(lattice_type, radius, period, eps_cyl, eps_med)
      # calculate the permittivity Fourier matrix:
    T = toeplitz2(crystal.calc_fourier(n_harm), n_harm, n_harm)

      # band diagram calculation parameters:
    k_points = np.zeros((ntot, 2), dtype=float) # points in k-space and two k-vector projections
    k_plot_axis = np.zeros((ntot)) # points along the k-space trajectory for plotting
    k_points[0,:] = psym[0][:]
    lsp = np.linspace(1,npt,npt)
    k_plot_axis[0] = dk = -np.sqrt((psym[1][0]-psym[0][0])**2 + (psym[1][1]-psym[0][1])**2)
    for sec in range(0,3): # section on the trajectory
        for cr in range(0,2): # x,y coordinate
            k_points[sec*npt+1:(sec+1)*npt+1,cr] = psym[sec][cr] + lsp * (psym[sec+1][cr]-psym[sec][cr])/npt
        k_plot_axis[sec*npt+1:(sec+1)*npt+1] = dk + \
            lsp * np.sqrt((k_points[sec*npt+2,0]-k_points[sec*npt+1,0])**2 + (k_points[sec*npt+2,1]-k_points[sec*npt+1,1])**2)
        dk += np.sqrt((psym[sec+1][0]-psym[sec][0])**2 + (psym[sec+1][1]-psym[sec][1])**2)

    data_band = np.zeros((n_band, ntot), dtype=float); # array to store the data
    for k in range(0, ntot):
        print(k, ntot)
          # set of the reciprocal lattice points:
        crystal.calc_reciprocal_lattice(n_harm, k_points[k,:])
        M = crystal.get_eig_mat(n_harm, polarization, T)
        eigval, eigvect = eig(M)
        eigval = (np.sqrt(eigval.flat))[np.real(eigval) > -1.0e-12]
        indices_sorted = np.argsort(np.real(eigval))
        W = eigval.flat[indices_sorted]
        data_band[:,k] = np.real(W[0:n_band])

      # plot the band diagram:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for k in range (0, n_band):
        plt.plot(k_plot_axis, data_band[k,:], color='blue')
    plt.xlabel(r"k-vector trajectory'")
    plt.ylabel(r"$\omega\Lambda/2\pi c$")
    xpoints = [-np.sqrt((psym[1][0]-psym[0][0])**2 + (psym[1][1]-psym[0][1])**2), \
               0,\
               np.sqrt((psym[2][0]-psym[1][0])**2 + (psym[2][1]-psym[1][1])**2),\
               np.sqrt((psym[2][0]-psym[1][0])**2 + (psym[2][1]-psym[1][1])**2) +\
               np.sqrt((psym[3][0]-psym[2][0])**2 + (psym[3][1]-psym[2][1])**2)]
    plt.xlim((xpoints[0], xpoints[3]))
    plt.ylim(bottom=0)
    plt.xticks(xpoints, symm_points)
    plt.show()

if __name__ == "__main__" and __package__ is None:
    plot_circular_PhC_dispersion_2D()