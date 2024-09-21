"""
    MIT Licence, © Alexey A. Shcherbakov, 2024
"""

import numpy as np
from typing import TypeVar

import planar as pw

T_scalar = TypeVar('T_scalar', float, complex, np.ndarray[float], np.ndarray[complex])
T_array = TypeVar('T_array', float, complex, np.ndarray[float], np.ndarray[complex])

def get_smatrix_layer(kh : float, kz : T_array, polarization : str) -> np.ndarray[complex]:
    n = kz.size
    if polarization == 's' or polarization == 'TE' or polarization == 'p' or polarization == 'TM':
        sm = np.zeros((2,2,n), dtype=complex)
        sm[0,1,:] = sm[1,0,:] = np.exp(1j * kh * kz)
    elif polarization == 'sp' or polarization == 'TEM':
        sm = np.zeros((2,2,2*n), dtype=complex)
        sm[0,1,0:n] = sm[0,1,n:2*n] = sm[1,0,0:n] = sm[1,0,n:2*n] = np.exp(1j * kh * kz)
    else:
        raise ValueError("Unknown polarization in smatrix.get_smatrix_layer")
    return sm

def get_smatrix_interface(eps1 : T_scalar, eps2 : T_scalar, kz1 : T_array, kz2 : T_array, polarization : str) -> np.ndarray[complex]:
    n = kz1.size
    if polarization == 'TE' or polarization == 's':
        sm = np.zeros((2,2,n), dtype=complex)
        sm[0,0,:] = ((kz1 - kz2) / (kz1 + kz2)).flat[:]
        sm[1,0,:] = 1.0 + sm[0,0,:]
        sm[1,1,:] = -sm[0,0,:]
        sm[0,1,:] = 1.0 + sm[1,1,:]
    elif polarization == 'TM' or polarization == 'p':
        sm = np.zeros((2,2,n), dtype=complex)
        sm[0,0,:] = ((eps2*kz1 - eps1*kz2) / (eps2*kz1 + eps1*kz2)).flat[:]
        sm[1,0,:] = 1.0 + sm[0,0,:]
        sm[1,1,:] = -sm[0,0,:]
        sm[0,1,:] = 1.0 + sm[1,1,:]
    elif polarization == 'TEM' or polarization == 'sp':
        sm = np.zeros((2,2,2*n), dtype=complex)
       	  # TE:
        sm[0,0,0:n] = ((kz1 - kz2) / (kz1 + kz2)).flat[:]
        sm[1,0,0:n] = 1.0 + sm[0,0,0:n]
        sm[1,1,0:n] = -sm[0,0,0:n]
        sm[0,1,0:n] = 1.0 + sm[1,1,0:n]
		  # TM:
        sm[0,0,n:2*n] = ((eps2*kz1 - eps1*kz2) / (eps2*kz1 + eps1*kz2)).flat[:]
        sm[1,0,n:2*n] = 1.0 + sm[0,0,n:2*n]
        sm[1,1,n:2*n] = -sm[0,0,n:2*n]
        sm[0,1,n:2*n] = 1.0 + sm[1,1,n:2*n]
    else:
        raise ValueError("Unknown polarization in smatrix.get_smatrix_interface")
    return sm
    
def multiply(sm1 : np.ndarray[complex], sm2 : np.ndarray[complex]) -> np.ndarray[complex]:
    if sm1.shape[-1] != sm2.shape[-1]:
        raise ValueError("Incompatible input size in smatrix.multiply")

    if np.size(sm1.shape) == 4 and np.size(sm2.shape) == 4: # both S-matrices are full
        if sm1.shape != sm2.shape:
            raise ValueError("Incompatible matrix size in smatrix.multiply")
        n = sm1.shape[-1]
        sm = np.zeros((2,2,n,n))

        tmp = -sm2[0,0,:,:] @ sm1[1,1,:,:]
        tmp.flat[::n+1] += 1.0
        tmp = (np.linalg.solve(tmp.T, sm1[0,1,:,:].T)).T                #TM = SM1(:,:,1,2) / TM;
        sm[0,1,:,:] = tmp @ sm2[0,1,:,:]                                #SM(:,:,1,2) = TM*SM2(:,:,1,2);
        sm[0,0,:,:] = sm1[0,0,:,:] + tmp @ sm2[0,0,:,:] @ sm1[1,0,:,:]  #SM(:,:,1,1) = SM1(:,:,1,1) + TM*SM2(:,:,1,1)*SM1(:,:,2,1);

        tmp = -sm1[1,1,:,:] @ sm2[0,0,:,:]                              # TM = -SM1(:,:,2,2)*SM2(:,:,1,1);
        tmp.flat[::n+1] += 1.0
        tmp = (np.linalg.solve(tmp.T, sm2[1,0,:,:].T)).T                # TM = SM2(:,:,2,1)/TM;
        sm[1,0,:,:] = tmp @ sm1[1,0,:,:]                                # SM(:,:,2,1) = TM*SM1(:,:,2,1);
        sm[1,1,:,:] = sm2[1,1,:,:] + tmp @ sm1[1,1,:,:] @ sm2[0,1,:,:]  # SM(:,:,2,2) = SM2(:,:,2,2) + TM*SM1(:,:,2,2)*SM2(:,:,1,2);

    elif np.size(sm1.shape) == 3 and np.size(sm2.shape) == 4: # first S-matrix is diagonal
        n = sm1.shape[-1]
        sm = np.zeros((2,2,n,n))

        tmp = -sm2[0,0,:,:] * sm1[1,1,:]                                # TM = -SM2(:,:,1,1).*transpose(SM1(:,2,2));
        tmp.flat[::n+1] += 1.0
        tmp = (np.linalg.solve(tmp.T, np.diag(sm1[0,1,:]))).T           # TM = diag(SM1(:,1,2))/TM;
        sm[0,1,:,:] = tmp @ sm2[0,1,:,:]                                # SM(:,:,1,2) = TM*SM2(:,:,1,2);
        sm[0,0,:,:] = tmp @ ( sm2[0,0,:,:] * sm1[1,0,:] )               # SM(:,:,1,1) = diag(SM1(:,1,1)) + TM*(SM2(:,:,1,1).*transpose(SM1(:,2,1)));
        sm[0,0,:,:].flat[::n+1] += sm1[0,0,:]

        tmp = -sm1[0,0,:].reshape(n,1) * sm2[0,0,:,:]                   # TM = -SM1(:,2,2).*SM2(:,:,1,1);
        tmp.flat[::n+1] += 1.0                                          # TM(1:n+1:end) = TM(1:n+1:end) + 1;
        tmp = (np.linalg.solve(tmp.T, sm2[1,0,:,:].T)).T                # TM = SM2(:,:,2,1) / TM;
        sm[1,0,:,:] = tmp * sm1[1,0,:]                                  # SM(:,:,2,1) = TM.*transpose(SM1(:,2,1));
        sm[1,1,:,:] = sm2[1,1,:,:] + tmp @ ( sm1[0,0,:].reshape(n,1) * sm2[0,1,:,:] ) # SM(:,:,2,2) = SM2(:,:,2,2) + TM*(SM1(:,2,2).*SM2(:,:,1,2));

    elif np.size(sm1.shape) == 4 and np.size(sm2.shape) == 3: # second S-matrix is diagonal
        n = sm1.shape[-1]
        sm = np.zeros((2,2,n,n))

        tmp = -sm2[0,0,:].reshape(n,1) * sm1[1,1,:,:]                   # TM = -SM2(:,1,1).*SM1(:,:,2,2);
        tmp.flat[::n+1] += 1.0                                          # TM(1:n+1:end) = TM(1:n+1:end) + 1;
        tmp = (np.linalg.solve(tmp.T, sm1[0,1,:,:].T)).T                # TM = SM1(:,:,1,2)/TM;
        sm[0,1,:,:] = tmp * sm2[0,1,:]                                  # SM(:,:,1,2) = TM.*transpose(SM2(:,1,2));
        sm[0,0,:,:] = sm1[0,0,:,:] + tmp @ ( sm2[0,0,:].reshape(n,1) * sm1[1,0,:,:]) # SM(:,:,1,1) = SM1(:,:,1,1) + TM*(SM2(:,1,1).*SM1(:,:,2,1));

        tmp = -sm1[1,1,:,:] * sm2[0,0,:]                                # TM = -SM1(:,:,2,2).*transpose(SM2(:,1,1));
        tmp.flat[::n+1] += 1.0                                          # TM(1:n+1:end) = TM(1:n+1:end) + 1;
        tmp = (np.linalg.solve(tmp.T, np.diag(sm2[1,0,:]))).T           # TM = diag(SM2(:,2,1)) / TM;
        sm[1,0,:,:] = tmp @ sm1[1,0,:,:]                                # SM(:,:,2,1) = TM*SM1(:,:,2,1);
        sm[1,1,:,:] = tmp @ ( sm1[1,1,:,:] * sm2[0,1,:] )               # SM(:,:,2,2) = diag(SM2(:,2,2)) + TM*(SM1(:,:,2,2).*transpose(SM2(:,1,2)));
        sm[1,1,:,:].flat[::n+1] += sm2[1,1,:]

    elif np.size(sm1.shape) == 3 and np.size(sm2.shape) == 3: # both S-matrix are diagonal
        n = sm1.shape[-1]
        sm = np.zeros((2,2,n), dtype=complex)

        tmp = 1.0 / ( 1.0 - sm1[1,1,:] * sm2[0,0,:] )                   # TM = SM1(:,1,2)./(1 - SM1(:,2,2).*SM2(:,1,1));
        sm[0,1,:] = sm1[0,1,:] * sm2[0,1,:] * tmp                       # SM(:,1,2) = SM2(:,1,2).*TM;
        sm[0,0,:] = sm1[0,0,:] + tmp * sm1[0,1,:] * sm2[0,0,:] * sm1[1,0,:] # SM(:,1,1) = SM1(:,1,1) + TM.*SM2(:,1,1).*SM1(:,2,1);
        sm[1,0,:] = sm2[1,0,:] * sm1[1,0,:] * tmp                       # SM(:,2,1) = SM1(:,2,1).*TM;
        sm[1,1,:] = sm2[1,1,:] + tmp * sm2[1,0,:] * sm1[1,1,:] * sm2[0,1,:] # SM(:,2,2) = SM2(:,2,2) + SM2(:,1,2).*SM1(:,2,2).*TM;
    
    return sm
