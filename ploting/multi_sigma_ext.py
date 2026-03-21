import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path


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

sigma_clean = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_e156149f7f78cbb6/sigma_ext.txt")
sigma = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_eaa359f7c9ec6818/sigma_ext.txt")

plt.plot(sigma_clean[:,0]*au_eV,
         sigma_clean[:,1] * au_nm**2,
         label="clean",
         color="#cc00cc",
         linewidth=1)

plt.plot(sigma[:,0]*au_eV,
         (sigma[:,1]*au_nm**2)/2,
         label="everything",
         color="#ff66ff",   # bright magenta
         linewidth=1)

plt.legend(loc='best', fontsize=22)
plt.xlabel('Energy (eV)', fontsize=12)
plt.savefig("sigma_ext_ZZ_rot_90.png", dpi=300, bbox_inches="tight")
plt.show()