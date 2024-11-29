"""
    MIT Licence, © Alexey A. Shcherbakov, 2024
"""
'''
Draw empty lattice approximation dispersion diagrams
for 2D square and hexagonal lattice photonic crystals
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

  # dispersion function for 2D empty lattice crystal:
def dispersion(kx : np.ndarray, ky: np.ndarray, m1 : int, m2 : int, K1 : float, K2 : float) -> np.ndarray:
    '''
    kx, ky: projections of the Bloch wavevector for a given set of Bloch wavevectors
    m1, m2: integers defining a particular band
    K1, K2: two-float lists defining the reciprocal lattice vectors
    '''
    return np.sqrt((kx + m1*K1[0] + m2*K2[0])**2 + (ky + m1*K1[1] + m2*K2[1])**2)

def plot_empty_lattice_dispersion_2D():
    period = 1 # period in length units
    nump = 20 # number of points in each segment of the dispersion diagram to plot

      # moduli of the reciprocal lattice vectors
    K_sqr = 2*np.pi/period # for the square lattice
    K_hex = 4*np.pi/period/np.sqrt(3) # for the hexagonal lattice

      # reciprocal lattice vectors
    K1_sqr = [K_sqr, 0]
    K2_sqr = [0, K_sqr]
    K1_hex = [0.5*K_hex, 0.5*np.sqrt(3)*K_hex]
    K2_hex = [0.5*K_hex, -0.5*np.sqrt(3)*K_hex]

    cw = 1/K_sqr # frequency normalization coefficient

      # plotting parameters
    plt.rcParams['text.usetex'] = True
    font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 22}
    matplotlib.rc('font', **font)
    #plt.figure(figsize=(20,10))

    fig, (ax_sqr, ax_hex) = plt.subplots(1, 2)

      # highlight symmetry points for the square lattice:
    ax_sqr.plot([0, 0], [0, 5], color="grey", linestyle='dashed') # Gamma-point
    ax_sqr.plot([0.5*K_sqr, 0.5*K_sqr], [0, 5], color="grey", linestyle='dashed') # X-point
      # highlight symmetry points for the hexagonal lattice:
    ax_hex.plot([0, 0], [0, 5], color="grey", linestyle='dashed') # Gamma-point
    ax_hex.plot([K_hex/np.sqrt(3), K_hex/np.sqrt(3)], [0, 5], color="grey", linestyle='dashed') # K-point

      # loop over the bands:
    for m1 in range(-3,3):
        for m2 in range(-3,3):
              # square lattice M->G line
            x = np.linspace(0, 0.5*np.sqrt(2)*K_sqr, nump) - 0.5*np.sqrt(2)*K_sqr
            y = cw * dispersion(np.linspace(0.5*K_sqr, 0, nump), 
                                np.linspace(0.5*K_sqr, 0, nump), 
                                m1, m2, K1_sqr, K2_sqr)
            ax_sqr.plot(x, y, color="blue")
              # square lattice G->X line
            x = np.linspace(0, 0.5*K_sqr, nump)
            y = cw * dispersion(np.linspace(0, 0.5*K_sqr, nump), 
                                0, m1, m2, K1_sqr, K2_sqr)
            ax_sqr.plot(x, y, color="blue")
            # square lattice X->M line
            x = 0.5*K_sqr + np.linspace(0, 0.5*K_sqr, nump)
            y = cw * dispersion(0.5*K_sqr, 
                                np.linspace(0, 0.5*K_sqr, nump), 
                                m1, m2, K1_sqr, K2_sqr)
            ax_sqr.plot(x, y, color="blue")
            
            # hexagonal lattice M->G line
            x = np.linspace(-0.5*K_hex, 0, nump)
            y = cw * dispersion(np.linspace(0.5*K_hex, 0, nump), 
                                0,
                                m1, m2, K1_hex, K2_hex)
            ax_hex.plot(x, y, color="blue")
            # hexagonal lattice G->K line
            x = np.linspace(0, K_hex/np.sqrt(3), nump)
            y = cw * dispersion(np.linspace(0, 0.5*K_hex, nump), 
                                np.linspace(0, 0.5*K_hex/np.sqrt(3), nump),
                                m1, m2, K1_hex, K2_hex)
            ax_hex.plot(x, y, color="blue")
            # hexagonal lattice K->M line
            x = np.linspace(0, 0.5*K_hex/np.sqrt(3), nump) + K_hex/np.sqrt(3)
            y = cw * dispersion(0.5*K_hex, 
                                np.linspace(0.5*K_hex/np.sqrt(3), 0, nump), 
                                m1, m2, K1_hex, K2_hex)
            ax_hex.plot(x, y, color="blue")

    ax_sqr.set_ylabel(r'$\omega\Lambda/2\pi c$')
    ax_sqr.set_xlim([-0.5*np.sqrt(2)*K_sqr, K_sqr])
    ax_sqr.set_ylim([0, 2])
    ax_sqr.set_xticks([0, -0.5*np.sqrt(2)*K_sqr, 0.5*K_sqr, K_sqr], [r'$\Gamma$', 'M', 'X', 'M'])
    ax_sqr.set_title('square lattice', pad = 20)

    ax_hex.set_xlim([-0.5*K_hex, 1.5*K_hex/np.sqrt(3)])
    ax_hex.set_ylim([0, 2])
    ax_hex.set_xticks([0, -0.5*K_hex, K_hex/np.sqrt(3), 1.5*K_hex/np.sqrt(3)], [r'$\Gamma$', 'M', 'K', 'M'])
    ax_hex.set_yticks([])
    ax_hex.set_title('hexagonal lattice', pad = 20)

    fig.set_figheight(10)
    fig.set_figwidth(20)

    plt.show()
    pass

if __name__ == "__main__" and __package__ is None:
    plot_empty_lattice_dispersion_2D()