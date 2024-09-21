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
    def __init__(self, lattice_type, radius, period, eps_cyl, eps_med) -> None:
        self.lattice_type = lattice_type
        self.radius = radius
        self.period = period
        self.eps_cyl = eps_cyl
        self.eps_med = eps_med

          # calculate the reciprocal lattice vector projections and moduli
        if lattice_type == "square":
            self.fill_fact = np.pi * radius^2 / period^2; # volume fraction occupied of cylinders
            self.K1 = np.array([1, 0])
            self.K2 = np.array([0, 1])
        elif lattice_type == "hexagonal":
            self.fill_fact = (2*np.pi/np.sqrt(3.0)) * radius^2 / period^2
            self.K1 = np.array([1, -1/np.sqrt(3.0)])
            self.K2 = np.array([0, 2/np.sqrt(3.0)])
        pass

    def calc_reciprocal_lattice(self, n_harm, k_bloch):
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
        TM = toeplitz2(FM, n_harm, n_harm)
        if polarization == 'TE' or polarization == 's':
            M = TM * self.Km * np.transpose(self.Km)
        elif polarization == 'TM' or polarization == 'p':
            M = TM * (self.Kx * np.transpose(self.Kx) + self.Ky * np.transpose(self.Ky))
        return M

#########################################################################################

  # parameters:
radius = 0.48 # radius of cylindrical rods
period = 1 # period (P > 2R)
eps_cyl = 1 # permittivity of cylinders
eps_med = 13; # permittivity of the surrounding medium
lattice_type = 'hexagonal'; # either "square" or "hexagonal"

polarization = 'TE' # % polarization
n_harm = 35 # number of Fourier harmonics in one dimension
k_bloch = [0.03, 0.3] # Bloch wavevector

  # band diagram calculation and visualization:
n_band = 5 # set the number of bands to store:
  
    # initialization of the band diagram calculation:
  # set numbers of points to calculate and coordinates of corner points in the reciprocal space
if lattice_type == 'square':
    n12 = 30; x1 = 0; y1 = 0 # Gamma point
    n23 = 30; x2 = 0.5; y2 = 0.5 # M point
    n31 = 30; x3 = 0.5; y3 = 0 # X point
elif lattice_type == 'hexagonal':
    n12 = 30; x1 = 0; y1 = 0 # Gamma point
    n23 = 30; x2 = 0.5; y2 = 0.5/np.sqrt(3) # M point
    n31 = 30; x3 = 2/3; y3 = 0 # K point

crystal = PhotonicCrystal2DCyl(lattice_type, radius, period, eps_cyl, eps_med)

  # band diagram calculation
nn = n12 + n23 + n31 # total number of points to evaluate
	# calculate the permittivity Fourier matrix:

crystal.calc_reciprocal_lattice(n_harm, k_bloch)
FM = crystal.calc_fourier(n_harm)

k_points = np.zeros([1, nn])
data_band = np.zeros([n_band, nn]); # array to store the data

  # line between points 1 and 2
for i in range(0, n12):
    k_bloch = np.array([(i-0.5)/n12 * (x2-x1) + x1, (i-0.5)/n12 * (y2-y1) + y1])
    crystal.calc_reciprocal_lattice(n_harm, k_bloch)
    M = crystal.get_eig_mat(n_harm, polarization, FM)
    eigval, eigvect = eig(M)
    eigenval = np.sqrt(eigenval)
    indices_sorted = np.argsort(np.real(eigenval))
    W = eigenval.flat[indices_sorted]
    k_points[i] = np.sqrt((k_bloch[0]-x1)^2 + (k_bloch[1]-y1)^2)
    data_band[:,i] = W[0:n_band-1]

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
