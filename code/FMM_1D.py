"""
    MIT Licence, © Alexey A. Shcherbakov, 2024
"""

#==================================================================================================

import numpy as np
from scipy.linalg import toeplitz, eig, inv, solve
from dataclasses import dataclass

#==================================================================================================

@dataclass
class DiffractionGratingProblem:
    """! Grating diffraction problem class
    
    Defines physical parameters of the problem. All length units [L] are supposed to be the same.
    """

    ## wavelength in [L], float
    wavelength: float = 1
    ## polarization, string being either 's' or 'p'
    polarization: str = 's'
    ## angle of incidence in degrees, float
    angle_incidence: float = 0
    kx_inc: float = 0
    # permittivities of media above an below the grating slab:
    epsilon_substrate: float = 1
    epsilon_superstrate: float = 1
    def __init__(self, wavelength: float, polarization: str, angle_incidence: float, \
                 epsilon_substrate: float, epsilon_superstrate: float) -> None:
        self.kx_inc = np.sqrt(epsilon_superstrate) * np.sin(angle_incidence * np.pi / 180.0)
        self.epsilon_substrate = epsilon_substrate
        self.epsilon_superstrate = epsilon_superstrate
        self.polarization = polarization
        self.wavelength = wavelength
        pass

#==================================================================================================

class PhotonicCrystalSlab():
    def __init__(self, period: float, fill_factors: np.array, epsilons: np.array, depth: float) -> None:
        self.period = period
        self.depth = depth
        if np.fabs(np.sum(fill_factors) - 1) > 1e-14:
            raise ValueError('initialization of PhotonicCrystalSlab: sum of fill_factors should be 1')
        if fill_factors.size != epsilons.size:
            raise ValueError('initialization of PhotonicCrystalSlab: fill_factors and epsilons arrays should be of the same size')
        num_elements = fill_factors.size
        self.fill_factors = np.reshape(fill_factors, (1, num_elements))
        self.epsilons = np.reshape(epsilons, (1, num_elements))
        pass

    def get_Fourier_epsilon(self, number_harmonics: int = 1):
        positions = -0.5 + 0.5 * self.fill_factors
        positions[0,1:] += np.cumsum(self.fill_factors[0,0:-1])

        indices = np.reshape(np.linspace(1, number_harmonics-1, number_harmonics-1), (number_harmonics-1, 1))

        eps_Fourier_vector = np.zeros((2, 2*number_harmonics-1), dtype=complex)
        tmp = np.sin(np.pi * indices * self.fill_factors) * np.exp((-2*np.pi*1j) * indices * positions) / (np.pi * indices)
        eps_Fourier_vector[0, number_harmonics:] = np.sum(self.epsilons * tmp, 1)
        eps_Fourier_vector[1, number_harmonics:] = np.sum((1/self.epsilons) * tmp, 1)
        eps_Fourier_vector[0, number_harmonics-2::-1] = np.sum(self.epsilons * np.conj(tmp), 1)
        eps_Fourier_vector[1, number_harmonics-2::-1] = np.sum((1/self.epsilons) * np.conj(tmp), 1)
        eps_Fourier_vector[0, number_harmonics-1] = np.sum(self.epsilons * self.fill_factors)
        eps_Fourier_vector[1, number_harmonics-1] = np.sum((1/self.epsilons) * self.fill_factors)

        eps_Fourier_matrix = toeplitz(eps_Fourier_vector[0, number_harmonics-1:], eps_Fourier_vector[0, number_harmonics-1::-1])
        ieps_Fourier_matrix = toeplitz(eps_Fourier_vector[1, number_harmonics-1:], eps_Fourier_vector[1, number_harmonics-1::-1])

        return eps_Fourier_matrix, ieps_Fourier_matrix

#==================================================================================================

class FourierMethod():
    def __init__(self, number_harmonics: int = 1) -> None:
        self.number_harmonics = number_harmonics
        self.central_harmonic = int((self.number_harmonics)/2) # index corresponding to the zero diffraction order
        pass

    def get_kx(self, kx_inc: float, Kg: float) -> np.ndarray:
        kx = kx_inc + Kg * ( \
            np.linspace(0,self.number_harmonics-1,self.number_harmonics) - self.central_harmonic )
        return kx

    def get_kz(self, kx: np.ndarray, epsilon: complex) -> np.ndarray:
        kz = np.sqrt(complex(epsilon) - kx**2)
        ind = np.angle(kz) < -1e-12
        kz[ind] = -kz[ind]
        return kz

    def get_smatrix_layer(self, kx: np.ndarray, kh: float, epsilon: complex) ->  np.ndarray:
        kz = self.get_kz(kx, epsilon)
        S = np.zeros((2, 2, self.number_harmonics, self.number_harmonics), dtype=complex)
        S[0,1,:,:].flat[0:self.number_harmonics+1:-1] = np.exp(1j * kh * kz)
        S[1,0,:,:] = S[0,1,:,:]
        return S

    def get_smatrix_interface(self, kx: np.ndarray, epsilon1: complex, epsilon2: complex, polarization: str) ->  np.ndarray:
        kz1 = self.get_kz(kx, epsilon1)
        kz2 = self.get_kz(kx, epsilon2)
        if polarization == 'p':
            kz1 /= epsilon1
            kz2 /= epsilon2
        S = np.zeros((2, 2, self.number_harmonics, self.number_harmonics), dtype=complex)
        S[0,0,:,:].flat[::self.number_harmonics+1] = (kz1 - kz2) / (kz1 + kz2)
        S[1,0,:,:].flat[::self.number_harmonics+1] = 1 + S[0,0,:,:].flat[::self.number_harmonics+1]
        S[1,1,:,:].flat[::self.number_harmonics+1] = -S[0,0,:,:].flat[::self.number_harmonics+1]
        S[0,1,:,:].flat[::self.number_harmonics+1] = 1 + S[1,1,:,:].flat[::self.number_harmonics+1]

        return S

    def composition_smatrices(self, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
        S = np.zeros((2, 2, self.number_harmonics, self.number_harmonics), dtype=complex)

        tmp = -S2[0,0,:,:] @ S1[1,1,:,:]
        tmp.flat[::self.number_harmonics+1] += 1.0
        tmp = (solve(tmp.T, S1[0,1,:,:].T)).T
        S[0,1,:,:] = tmp @ S2[0,1,:,:]
        S[0,0,:,:] = S1[0,0,:,:] + tmp @ S2[0,0,:,:] @ S1[1,0,:,:]

        tmp = -S1[1,1,:,:] @ S2[0,0,:,:]
        tmp.flat[::self.number_harmonics+1] += 1.0
        tmp = (np.linalg.solve(tmp.T, S2[1,0,:,:].T)).T
        S[1,0,:,:] = tmp @ S1[1,0,:,:]
        S[1,1,:,:] = S2[1,1,:,:] + tmp @ S1[1,1,:,:] @ S2[0,1,:,:]
        
        return S
    
    def diffract(self, S: np.ndarray, V_in: np.ndarray) -> np.ndarray:
        V_out = np.zeros_like(V_in, dtype=complex)
        V_out[0,:] = S[0,0,:,:] @ V_in[0,:] + S[0,1,:,:] @ V_in[1,:]
        V_out[1,:] = S[1,0,:,:] @ V_in[0,:] + S[1,1,:,:] @ V_in[1,:]
        return V_out

    def plane_wave(self, direction: str) -> np.ndarray:
        V_pw = np.zeros((2, self.number_harmonics), dtype=complex)
        if direction == 'from above':
            V_pw[1, self.central_harmonic] = 1.0
        elif direction == 'from below':
            V_pw[1, self.central_harmonic] = 1.0
        else:
            raise ValueError('FourierMethod.plane_wave: unknown direction')
        return V_pw
    
    def get_efficiencies(self, V_in: np.ndarray, V_out: np.ndarray, \
                         problem: DiffractionGratingProblem, slab: PhotonicCrystalSlab) -> np.ndarray:
        V_eff = np.zeros_like(V_out, dtype=float)

        kx = self.get_kx(problem.kx_inc, problem.wavelength / slab.period)
        kz_sub = self.get_kz(kx, problem.epsilon_substrate)
        kz_sup = self.get_kz(kx, problem.epsilon_superstrate)

        if problem.polarization == 's':
            P_in = np.sum( np.abs(V_in[0,:]**2) * np.real(kz_sub[:]) \
                           + np.abs(V_in[1,:]**2) * np.real(kz_sup[:]) )
            V_eff[0,:] = np.abs(V_out[0,:]**2) * (kz_sub[:]).real
            V_eff[1,:] = np.abs(V_out[1,:]**2) * (kz_sup[:]).real
        elif problem.polarization == 'p':
            P_in = np.sum( np.abs(V_in[0,:]**2) * np.real(kz_sub[:] / problem.epsilon_substrate) \
                           + np.abs(V_in[1,:]**2) * np.real(kz_sup[:] / problem.epsilon_superstrate) )
            V_eff[0,:] = np.abs(V_out[0,:]**2) * (kz_sub[:] / problem.epsilon_substrate).real
            V_eff[1,:] = np.abs(V_out[1,:]**2) * (kz_sup[:] / problem.epsilon_superstrate).real

        return (V_eff / P_in)

#==================================================================================================

class FourierModalMethod(FourierMethod):
    def __init__(self, number_harmonics: int = 1) -> None:
        super().__init__(number_harmonics)
        pass

    def get_smatrix_phc_slab(self, problem: DiffractionGratingProblem, slab: PhotonicCrystalSlab) -> np.ndarray:
        kx = self.get_kx(problem.kx_inc, problem.wavelength / slab.period)
          # Toeplitz Fourier matrices of the permittivity and inverse permittivity functions:
        eps_Fourier_matrix, ieps_Fourier_matrix = slab.get_Fourier_epsilon(self.number_harmonics)
          # calculate eigenstates for the TE polarization:
        if problem.polarization == 's':
            E_eig = eps_Fourier_matrix
            E_eig.flat[::self.number_harmonics+1] -= (kx**2).flat[:]
              # eigenvalue problem for the electric field:
            betas, E_eig = eig(E_eig, overwrite_a=True)
            betas = np.reshape(betas, (1, self.number_harmonics))
              # sort the eigenvalues and eigenvectors:
            indices_sorted = np.argsort(-np.real(betas))
            betas.flat[:] = betas.flat[indices_sorted]
            E_eig[:,:] = E_eig[:,indices_sorted.flat]
              # get propagation constants and check that they lie in the upper complex half-plane
            betas = np.sqrt(betas)
            indices_neg = np.angle(betas) < -1e-14
            betas[indices_neg] = -betas[indices_neg]
              # magnetic field eigenvectors:
            H_eig = -E_eig * betas
          # calculate eigenstates for the TM polarization:
        elif problem.polarization == 'p':
            M = inv(ieps_Fourier_matrix)
            E_eig = -(solve(eps_Fourier_matrix.T, np.diag(kx))).T * kx
            E_eig.flat[::self.number_harmonics+1] += 1
              # eigenvalue problem for the electric field:
            betas, E_eig = eig(E_eig @ M)
            betas = np.reshape(betas, (1, self.number_harmonics))
              # sort the eigenvalues and eigenvectors:
            indices_sorted = np.argsort(-np.real(betas))
            betas.flat[:] = betas.flat[indices_sorted]
            E_eig[:,:] = E_eig[:,indices_sorted.flat]
              # get propagation constants and check that they lie in the upper complex half-plane
            betas = np.sqrt(betas)
            indices_neg = np.angle(betas) < -1e-14
            betas[indices_neg] = -betas[indices_neg]
              # magnetic field eigenvectors:
            H_eig = (M @ E_eig) / betas
        else:
            raise ValueError('in FourierModalMethod.grating_smatrix: unknown polarization')
          # get interface T-matrices:
        T_sub = self.phc_interface_tmatrix(problem, E_eig, H_eig, kx, problem.epsilon_substrate)
        T_sup = self.phc_interface_tmatrix(problem, E_eig, H_eig, kx, problem.epsilon_superstrate)
          # calculate the S-matrix:
        kh = 2 * np.pi * slab.depth / problem.wavelength
        betas = np.exp(1j * kh * betas)

        M1 = np.zeros((2*self.number_harmonics, 2*self.number_harmonics), dtype=complex)
        M2 = np.zeros((2*self.number_harmonics, 2*self.number_harmonics), dtype=complex)

        ind1 = slice(0, self.number_harmonics, 1)
        ind2 = slice(self.number_harmonics, 2*self.number_harmonics, 1)

        M1[ind1,ind1] = T_sub[1,0,:,:]
        M1[ind1,ind2] = T_sub[1,1,:,:] * betas
        M1[ind2,ind1] = T_sup[0,0,:,:] * betas
        M1[ind2,ind2] = T_sup[0,1,:,:]
        M2[ind1,ind1] = T_sub[0,0,:,:]
        M2[ind1,ind2] = T_sub[0,1,:,:] * betas
        M2[ind2,ind1] = T_sup[1,0,:,:] * betas
        M2[ind2,ind2] = T_sup[1,1,:,:]
        M1 = (solve(M2.T, M1.T)).T
          # fill S-matrix
        S = np.zeros((2, 2, self.number_harmonics, self.number_harmonics), dtype=complex)
        S[0,0,ind1,ind1] = M1[ind1,ind1]
        S[0,1,ind1,ind1] = M1[ind1,ind2]
        S[1,0,ind1,ind1] = M1[ind2,ind1]
        S[1,1,ind1,ind1] = M1[ind2,ind2]

        return S

    def phc_interface_tmatrix(self, problem: DiffractionGratingProblem, E_eig: np.ndarray, H_eig: \
                              np.ndarray, kx: np.ndarray, epsilon: complex) -> np.ndarray:
        T = np.zeros((2, 2, self.number_harmonics, self.number_harmonics), dtype=complex)
        kz = self.get_kz(kx, epsilon)

        if problem.polarization == 's':
            ikz = np.reshape(0.5/kz, (self.number_harmonics, 1))
            T[0,0,:,:] = -H_eig * ikz
            T[1,0,:,:] = 0.5 * E_eig - T[0,0,:,:]
            T[0,0,:,:] += 0.5 * E_eig
            T[0,1,:,:] = T[1,0,:,:]
            T[1,1,:,:] = T[0,0,:,:]
        elif problem.polarization == 'p':
            eikz = np.reshape((0.5*epsilon)/kz, (self.number_harmonics, 1))
            T[0,0,:,:] = E_eig * eikz
            T[1,0,:,:] = 0.5 * H_eig - T[0,0,:,:]
            T[0,0,:,:] += 0.5 * H_eig
            T[0,1,:,:] = -T[1,0,:,:]
            T[1,1,:,:] = -T[0,0,:,:]
        else:
            raise ValueError('in FourierModalMethod.grating_smatrix: unknown polarization')
        return T

###################################################################################################

if __name__ == "__main__" and __package__ is None:
    wavelength = 0.632
    polarization = 'p'
    angle_incidence = 15
    epsilon_substrate = 2.25
    epsilon_superstrate = 1

    problem = DiffractionGratingProblem(wavelength, polarization, angle_incidence, epsilon_substrate, epsilon_superstrate)

    number_harmonics = 15
    fmm = FourierModalMethod(number_harmonics)

    period = 0.8
    fill_factors = np.array([0.4, 0.6], dtype=float)
    phc_epsilons = np.array([1, 12], dtype=complex)
    slab_thickness = 0.4
    grating = PhotonicCrystalSlab(period, fill_factors, phc_epsilons, slab_thickness)

    # calculation of diffraction efficiencies:
    S = fmm.get_smatrix_phc_slab(problem, grating)
    V_in = fmm.plane_wave('from above')
    V_out = fmm.diffract(S, V_in)
    V_eff = fmm.get_efficiencies(V_in, V_out, problem, grating)
    print(V_eff.T)
    print(np.sum(V_eff))


