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



#============================clean HHG=============================
dipole_acc_data_clean = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_armchair_71ab3cf56b148504/dipole_acc.txt")
omega_eV = dipole_acc_data_clean[:,0] * au_eV
dipole_acc_data_clean = dipole_acc_data_clean[:, 1] + 1j * dipole_acc_data_clean[:, 2]  # shape (N,)
y_axis = np.abs(dipole_acc_data_clean)**2

# Find resonance frequency (maximum intensity)
index1 = np.argmax(y_axis)
omega_01 = omega_eV[index1]

# Normalize axes 
x_val1 = omega_eV / omega_01
y_val1 = y_axis / y_axis[index1]

#===============HHG with everything=================================

dipole_acc_data = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_armchair_da1ae96c6aca1cfd/dipole_acc.txt")
omega_eV = dipole_acc_data[:,0] * au_eV
dipole_acc_data = dipole_acc_data[:, 1] + 1j * dipole_acc_data[:, 2]  # shape (N,)
y_axis = np.abs(dipole_acc_data)**2


# Find resonance frequency (maximum intensity)
index = np.argmax(y_axis)
omega_0 = omega_eV[index]

# Normalize axes 
x_val = omega_eV / omega_0
y_val = y_axis / y_axis[index]



plt.rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x_val1, y_val1, linewidth=1, label = "Clean HHG", color= "k")
ax.plot(x_val, y_val, linewidth = 1, label = "Everything", color= "m")

ax.set_yscale("log")
ax.set_xlabel(r'$\hbar\omega /\omega_0 $', fontsize=24)
ax.set_ylabel(r'$|\ddot{p}(\omega)|^2 / |\ddot{p}(\omega_0)|^2$', fontsize=24)

ax.tick_params(labelsize=12)
ax.set_xlim(0, 30)

ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth=0.8)
ax.grid(which='minor', linestyle=':', linewidth=0.5)
ax.tick_params(axis='x', which='minor', bottom=True)

ax.legend(loc='best', fontsize=22)
plt.tight_layout()
plt.savefig("HHG_comparison_armchair.png", dpi=300, bbox_inches="tight")
plt.show()
