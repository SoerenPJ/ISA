import numpy as np
import matplotlib.pyplot as plt
import os 
# constants
au_eV = 27.2114
au_nm = 0.0529177
au_s = 2.41888e-17
au_fs = au_s * 1e15
au_c = 137.036
au_kg = 9.10938e-31
au_kB = 3.16681e-6
au_hbar = 1.0
au_m = 1.0
au_J = 4.35974e-18
au_w = 4.13414e16
au_I = 3.50945e16
au_me = 9.10938e-31
alpha = 1.0 / au_c
e = 1
au_mu_0 = 4 * np.pi * 1e-7 * (au_eV * au_s) / (au_J * au_m)


a = 0.142 / au_nm           #Lattice constant 

mu = 0.0/au_eV                     # Chemical potential
#Physical parameters
N = 50
T = 100
t1 = -2.8/au_eV
t2 = t1;#// + 0.5

#spin_on = false
#two_dim = false


# external field parameters
Intensity = (1e13/au_kg) * (au_s*au_s*au_s)  #// time pules = 1e13 ddf = 1e15
E0 = np.sqrt((2 *np.pi* Intensity) / (au_c)) #// Electric field amplitude

t_shift = 200 / au_fs
sigma_gaus = 60.07/au_fs 
sigma_ddf = 0.01/au_fs

gamma = 0.01/au_eV
au_omega = 0.2/au_eV
au_omega_fourier = 0.0 




out_dir = "Ploting/base_plots"
os.makedirs(out_dir, exist_ok=True)

eigenvalues = np.loadtxt('eigenvalues.txt')
rho_j_space = np.loadtxt('rho0_j_space.txt')
#reshapet the rho_j_space to a NxN matrix
rho_j_NxN = rho_j_space[:,0::2] + 1j*rho_j_space[:,1::2]
#generate the number of sites array
N_sites = np.arange(rho_j_NxN.shape[1])


rho_l_space = np.loadtxt('rho0_l_space.txt')
#reshape the rho_l_space to a NxN matrix
rho_l_NxN = rho_l_space[:,0::2] + 1j*rho_l_space[:,1::2]

dipole_moments = np.loadtxt('dipole_time_evolution.txt')
dipole = np.real(dipole_moments[:,1])
time = np.real(dipole_moments[:,0]*au_fs)


print("shape of dipolemoments", dipole_moments.shape)

#exit()
def baseplots(N_sites, t, eigenvalues, rho_j_NxN, rho_l_NxN, dipole):#, rho_l_space, dipole_moments):
    
    # Plot of eigenvalues
    plt.figure(figsize=(8, 6))
    plt.plot(eigenvalues[:,0] *au_eV, 'o')
    plt.title('Eigenvalues ')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue [eV]')
    plt.grid()
    plt.savefig(os.path.join(out_dir, "eigenvalues_plot.png"), dpi=300, bbox_inches="tight")

    #plot of rho_j_space
    plt.figure(figsize = (8,6))
    # extrar onyl the diagonal of the rho_j_space
    diag_rho_j = np.diag(rho_j_NxN)
    plt.plot(N_sites,diag_rho_j, '-o')
    plt.title('Density matrix diagonal elements in site space')
    plt.xlabel('Site index')
    plt.ylabel(r'$\rho_{jj}$')
    plt.grid()
    plt.savefig(os.path.join(out_dir, "rho_j_space_diagonal_plot.png"), dpi=300, bbox_inches="tight")

    #plot of the rho_l_space
    rho_l_diag = np.diag(rho_l_NxN)
    plt.figure(figsize = (8,6))
    plt.plot(N_sites,rho_l_diag, '-o')
    plt.title('Density matrix diagonal elements in lattice space')
    plt.xlabel('Site index')
    plt.ylabel(r'$\rho_{jj}$')
    plt.grid()
    plt.savefig(os.path.join(out_dir, "rho_l_space_diagonal_plot.png"), dpi=300, bbox_inches="tight")
    # Plot of dipole moment time evolution
    plt.figure(figsize=(8,6))
    plt.plot(time, dipole)
    plt.xlabel('Time [fs]')
    plt.ylabel("Dipole moment [au]")
    plt.title("Time evolution of dipole moment")
    plt.savefig(os.path.join(out_dir, "Dipole_time_evolution_plot.png"), dpi= 300, bbox_inches="tight")
    plt.show()
    
    
    plt.close()

baseplots(N_sites, time, eigenvalues, rho_j_NxN, rho_l_NxN, dipole)