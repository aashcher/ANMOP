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
def dispersion(kx, ky, m1, m2, K1, K2):
    return np.sqrt((kx + m1*K1[0] + m2*K2[0])**2 + (ky + m1*K1[1] + m2*K2[1])**2)

def plot_empty_lattice_dispersion():
    period = 1
    nump = 20 # number of points in each segment

    K_sqr = 2*np.pi/period
    K_hex = 4*np.pi/period/np.sqrt(3)

    K1_sqr = [K_sqr, 0]
    K2_sqr = [0, K_sqr]
    K1_hex = [0.5*K_hex, 0.5*np.sqrt(3)*K_hex]
    K2_hex = [0.5*K_hex, -0.5*np.sqrt(3)*K_hex]

    cw = 1/K_sqr # frequency normalization coefficient

    plt.rcParams['text.usetex'] = True
    font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 22}
    matplotlib.rc('font', **font)
    #plt.figure(figsize=(20,10))

    fig, (ax_sqr, ax_hex) = plt.subplots(1, 2)

    ax_sqr.plot([0, 0], [0, 5], color="grey", linestyle='dashed') # Gamma-point
    ax_sqr.plot([0.5*K_sqr, 0.5*K_sqr], [0, 5], color="grey", linestyle='dashed') # X-point

    ax_hex.plot([0, 0], [0, 5], color="grey", linestyle='dashed') # Gamma-point
    ax_hex.plot([K_hex/np.sqrt(3), K_hex/np.sqrt(3)], [0, 5], color="grey", linestyle='dashed') # K-point

    for m1 in range(-3,3):
        for m2 in range(-3,3):
            # square MG line
            x = np.linspace(0, 0.5*np.sqrt(2)*K_sqr, nump) - 0.5*np.sqrt(2)*K_sqr
            y = cw * dispersion(np.linspace(0.5*K_sqr, 0, nump), 
                                np.linspace(0.5*K_sqr, 0, nump), 
                                m1, m2, K1_sqr, K2_sqr)
            ax_sqr.plot(x, y, color="blue")
            # square GX line
            x = np.linspace(0, 0.5*K_sqr, nump)
            y = cw * dispersion(np.linspace(0, 0.5*K_sqr, nump), 
                                0, m1, m2, K1_sqr, K2_sqr)
            ax_sqr.plot(x, y, color="blue")
            # square XM line
            x = 0.5*K_sqr + np.linspace(0, 0.5*K_sqr, nump)
            y = cw * dispersion(0.5*K_sqr, 
                                np.linspace(0, 0.5*K_sqr, nump), 
                                m1, m2, K1_sqr, K2_sqr)
            ax_sqr.plot(x, y, color="blue")
            
            # hexagonal MG line
            x = np.linspace(-0.5*K_hex, 0, nump)
            y = cw * dispersion(np.linspace(0.5*K_hex, 0, nump), 
                                0,
                                m1, m2, K1_hex, K2_hex)
            ax_hex.plot(x, y, color="blue")
            # hexagonal GK line
            x = np.linspace(0, K_hex/np.sqrt(3), nump)
            y = cw * dispersion(np.linspace(0, 0.5*K_hex, nump), 
                                np.linspace(0, 0.5*K_hex/np.sqrt(3), nump),
                                m1, m2, K1_hex, K2_hex)
            ax_hex.plot(x, y, color="blue")
            # hexagonal KM line
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

if __name__ == "__main__" and __package__ is None:
    plot_empty_lattice_dispersion()