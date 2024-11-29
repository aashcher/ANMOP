"""
    MIT Licence, © Alexey A. Shcherbakov, 2024
"""
import numpy as np
from scipy.linalg import toeplitz, eig
import matplotlib.pyplot as plt

  # calculate a 2D Toeplitz matrix from a 2D Fourier transform vector
def toeplitz2(M, nx, ny):
    CT = np.zeros([2*nx-1, 2*ny-1, 2*ny-1])
    for i in range(0, 2*nx-1):
        CT[i,:,:] = toeplitz(M[i,ny-1:],M[i,ny-1::-1])
    T = np.zeros([(2*nx-1)*(2*ny-1), (2*nx-1)*(2*ny-1)])
    for i in range(0, nx):
        for j in range(0, nx):
            T[i*ny:(i+1)*ny,j*ny:(j+1)*ny] = CT[nx+j-i,:,:]
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
            self.fill_fact = np.pi * radius^2 / period^2; # volume fraction occupied by the cylinders
            self.K1 = np.array([1, 0])
            self.K2 = np.array([0, 1])
        elif lattice_type == "hexagonal":
            self.fill_fact = (2*np.pi/np.sqrt(3.0)) * radius^2 / period^2 # volume fraction occupied by the cylinders
            self.K1 = np.array([1, -1/np.sqrt(3.0)])
            self.K2 = np.array([0, 2/np.sqrt(3.0)])
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
        #self.Kx = self.Kx.reshape(1, [])
        #self.Ky = reshape(lattice.Ky, 1, [])
        #self.Km = reshape(lattice.Km, 1, [])
        pass

    def calc_fourier(self, n_harm):
        '''
        calculate the Fourier amplitude vector of the inverse dielectric permittivitty periodic coordinate function
        h_harm: total number of Fourier harmonics to calculate in one dimension
        '''
        rp = self.radius / self.period

        FM = np.zeros([2*n_harm-1, 2*n_harm-1], dtype=complex)

        I1, I2 = np.meshgrid(np.linspace(0, n_harm-1, n_harm), np.linspace(0, n_harm-1, n_harm))	
        tF = 2*np.pi*rp*np.sqrt((self.K1(1)*I1 + self.K2(1)*I2)**2 + (self.K1(2)*I1 + self.K2(2)*I2)**2);
        ind = abs(tF) > 1.0e-14
        tF[ind] = 2 * np.besselj(1,tF(ind)) / tF(ind)
        tF[~ind] = 1.0
        FM[n_harm-1:,n_harm-1:] = tF
        FM[n_harm::-1,n_harm::-1] = tF

        tF = 2*np.pi*rp*np.sqrt((self.K1(1)*I1 - self.K2(1)*I2)**2 + (self.K1(2)*I1 - self.K2(2)*I2)**2)
        ind = abs(tF) > 1.0e-14
        tF[ind] = 2 * np.besselj(1,tF(ind)) / tF(ind)
        tF[~ind] = 1.0
        FM[n_harm:,n_harm-2::-1] = tF[1:,1:]
        FM[n_harm-2::-1,n_harm:] = tF[1:,1:]
        
        FM *= self.fill_fact * (1/self.eps_cyl - 1/self.eps_med)
        FM(n_harm-1, n_harm-1) += 1.0/self.eps_med

        return FM

    def get_eig_mat(self, n_harm, polarization, FM):
        '''
        fill the matrix for the 2D photonic crystal eigenvalue problem
        h_harm: total number of Fourier harmonics to calculate in one dimension
        polarization: 'TE', 's' or 'TM', 'p'
        FM: calculate the 2D Fourier amplitude vector of the inverse dielectric permittivitty
        '''
        TM = toeplitz2(FM, n_harm, n_harm)
        if polarization == 'TE' or polarization == 's':
            M = TM * self.Km * np.transpose(self.Km)
        elif polarization == 'TM' or polarization == 'p':
            M = TM * (self.Kx * np.transpose(self.Kx) + self.Ky * np.transpose(self.Ky))
        return M

#########################################################################################

def plot_circular_PhC_dispersion_2D():
      # parameters:
    radius = 0.48 # radius of cylindrical rods
    period = 1 # period (P > 2R)
    eps_cyl = 1 # permittivity of cylinders
    eps_med = 13; # permittivity of the surrounding medium
    lattice_type = 'hexagonal'; # either "square" or "hexagonal"

    polarization = 'TE' # % polarization
    n_harm = 15 # number of Fourier harmonics in one dimension

    n_band = 5 # set the number of bands to calculate:
  
      # initialization of the band diagram calculation:
    np = 30 # number of points in each segment of the dispersion diagram to plot
    ntot = 3*np+1 # total number of points
      # symmetry points:
    if lattice_type == 'square':
        psym = [[0.0, 0.0], [0.5, 0.5], [0.5, 0.0]] # Gamma -> M -> X
    elif lattice_type == 'hexagonal':
        psym = [[0.0, 0.0], [0.5, 0.5/np.sqrt(3.0)], [2.0/3.0, 0.0]] # Gamma -> M -> K

      # initialize the crystal class
    crystal = PhotonicCrystal2DCyl(lattice_type, radius, period, eps_cyl, eps_med)
      # set of the reciprocal lattice points:
    crystal.calc_reciprocal_lattice(n_harm, k_bloch)
      # calculate the permittivity Fourier matrix:
    FM = crystal.calc_fourier(n_harm)

      # band diagram calculation parameters:
    k_points = np.zeros([ntot, 2])
    for ax in range(0,1):
        k_points[0:np+1,ax] = psym[0][ax] + np.linspace(0,np,np+1) * (psym[1][ax]-psym[0][ax])/np
        k_points[np+1:2*np+1,ax] = psym[1][ax] + np.linspace(1,np,np) * (psym[2][ax]-psym[1][ax])/np
        k_points[2*np+1:3*np+1,ax] = psym[2][ax] + np.linspace(1,np,np) * (psym[0][ax]-psym[2][ax])/np

    data_band = np.zeros([n_band, ntot]); # array to store the data
    for k in range(0, ntot):
        crystal.calc_reciprocal_lattice(n_harm, k_points[k,:])
        M = crystal.get_eig_mat(n_harm, polarization, FM)
        eigval, eigvect = eig(M)
        eigenval = np.sqrt(eigenval)
        indices_sorted = np.argsort(np.real(eigenval))
        W = eigenval.flat[indices_sorted]
        data_band[:,k] = W[0:n_band]
    
    k_axis = np.sqrt(np.sum(k_points[:,:]**2, axis=-1))

delta_k = np.sqrt((x2-x1)^2 + (y2-y1)^2); # k-line shift

  # line between points 2 and 3
for i in range(0, n23):
    k_bloch = np.array([((i-0.5)/n23) * (x3-x2) + x2, ((i-0.5)/n23) * (y3-y2) + y2])
    crystal.calc_reciprocal_lattice(n_harm, k_bloch)
    M = crystal.get_eig_mat(n_harm, polarization, FM)
    eigval, eigvect = eig(M)
    eigenval = np.sqrt(eigenval)
    indices_sorted = np.argsort(np.real(eigenval))
    W = eigenval.flat[indices_sorted]
    k_points[n12+i] = delta_k + np.sqrt((k_bloch[0]-x2)^2 + (k_bloch[1]-y2)^2)
    data_band[:, n12+i] = W[0:n_band-1]

delta_k = delta_k + np.sqrt((x3-x2)^2 + (y3-y2)^2)

  # line between points 3 and 1
for i in range(0, n31):
    k_bloch = np.array([((i-0.5)/n31) * (x1-x3) + x3, ((i-0.5)/n31) * (y1-y3) + y3])
    crystal.calc_reciprocal_lattice(n_harm, k_bloch)
    M = crystal.get_eig_mat(n_harm, polarization, FM)
    eigval, eigvect = eig(M)
    eigenval = np.sqrt(eigenval)
    indices_sorted = np.argsort(np.real(eigenval))
    W = eigenval.flat[indices_sorted]
    k_points[n12+n23+i] = delta_k + np.sqrt((k_bloch[0]-x3)^2 + (k_bloch[1]-y3)^2);
    data_band[:,n12+n23+i] = W[0:n_band-1]

  ## plot the band diagram:

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for k in range (0, n_band):
    plt.plot(k_points, data_band[k,:], color='blue')
plt.xlabel(r"k-vector trajectory'")
plt.ylabel(r"$\omega\Lambda/2\pi c$")
plt.show()
plt.xticks([0, (np.sqrt((x2-x1)^2+(y2-y1)^2)), delta_k, (delta_k+sqrt((x1-x3)^2 + (y1-y3)^2))], \
                ['\Gamma','M','X/K','\Gamma'])

if __name__ == "__main__" and __package__ is None:
    plot_circular_PhC_dispersion_2D()