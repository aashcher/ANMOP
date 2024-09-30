[![en](https://img.shields.io/badge/lang-EN-blue.svg)](https://github.com/aashcher/ANMOP/blob/main/README.md)
[![ru](https://img.shields.io/badge/lang-RU-green.svg)](https://github.com/aashcher/ANMOP/blob/main/README.ru.md)

# Analytical and Numerical Methods in Optics and Photonics Course

When analyzing electrodynamic processes in problems with complex boundaries and inhomogeneous and/or anisotropic spatial distribution of matter, one has to resort to the use of numerical methods. On the one hand, to solve such problems one can use the general most widely spread approaches to the solution of differential equations, such as finite element methods (FEM) or finite difference method (FDM), but a large enough range of problems, both scientific and engineering, can be solved with the help of more specialized approaches, which in the field of their applicability are much more effective than the mentioned FEM and FDM. This course is devoted to familiarization with such approaches and examples of their application for solving engineering and physical problems in the field of optics and photonics.

The course is divided into blocks that can be studied relatively independently of each other. The basic facts common to all blocks are presented in the general part. In this sense, the course is adaptive and makes it possible to modify it to meet the needs of the audience. For the moment the course consists of five sections: general, and sections devoted to photonic crystals, diffraction gratings and metasurfaces, scattering structures and solution of inverse and optimization problems. Each section contains a theoretical part, accompanying demonstration codes in Python, and exercises. There are three tracks within the exercises: engineering, computational, and theoretical. The exercises in the engineering track focus on practicing the use of off-the-shelf codes to solve application problems. In the exercises of the computational track more emphasis is placed on the development of programs to perform calculations. The theoretical track is dominated by tasks for working through the theoretical material.

## Sections

### Common Part

1. [Electromagnetic Waves](https://nbviewer.org/github/aashcher/ANMOP/blob/main/nb_en/Common%20part%201.%20Electromagnetic%20waves.ipynb)
2. [S- and T-Matrix Methods](https://nbviewer.org/github/aashcher/ANMOP/blob/main/nb_en/Common%20part%202.%20S-%20and%20Т-matrix%20methods.ipynb)
3. [Green's Functions](https://nbviewer.org/github/aashcher/ANMOP/blob/main/nb_en/Common%20part%203.%20Greens%20functions%20of%20the%20Helholtz%20equation.ipynb)
4. [Resonnces in Open Systems](https://nbviewer.org/github/aashcher/ANMOP/blob/main/nb_en/Common%20part%204.%20Resonances%20in%20open%20systems.ipynb)
5. [Methods of Calculation of S-matrix Poles]()
6. [Tasks](https://nbviewer.org/github/aashcher/ANMOP/blob/main/nb_en/Common%20part.%20Tasks.ipynb)

### Photonic Crystals

1. Photonic Crystals: Prperties and Applications
2. Modal Basis for 1D Crystals
3. Fourier Space Methods
4. Tasks

### Diffraction Gratings and Metasurfaces

1. Gratings: Properties and Applications
2. Modal Methods
3. Resonant Properties
4. Tasks

### Electromagnetic Scattering

1. Scattering of Electromagnetic Waves: Overview
2. The Discarete Dipole Approximation and Method of Moments
3. Waterman T-matrix and the Extended Boundary Condition Method
4. Multiple Scattering
5. Tasks

### Inverse and Optimization Problems

1. Gradient-Based Optimization and the Adjoint Method
2. Stochastic Optimization
3. Deep Learning Approaches
4. Tasks
