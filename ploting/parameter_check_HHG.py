import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import FixedLocator
au_eV = 27.2114


tau_50_fast =    np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_fa3cc2263192dfbd/dipole_acc.txt")
tau_50_medium =  np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_6250b57ea76fa0ba/dipole_acc.txt")
tau_50_slow =    np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_215c1d019cbbf746/dipole_acc.txt")
tau_50_extreme = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_f38a417371c3a502/dipole_acc.txt")


tau_10_fast =   np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_c393772003220936/dipole_acc.txt")
tau_10_medium = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_790f7f1eff52a737/dipole_acc.txt")
tau_10_slow =   np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_ffffce33119fbdc3/dipole_acc.txt")
tau_10_extreme = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_cce765dddb153d7/dipole_acc.txt")

omega_eV_tau_50_fast = tau_50_fast[:,0]*au_eV
dipole_acc_tau_50_fast = tau_50_fast[:, 1] + 1j * tau_50_fast[:, 2]  
y_axis_tau_50_fast = np.abs(dipole_acc_tau_50_fast)**2


omega_eV_tau_50_medium = tau_50_medium[:,0]*au_eV
dipole_acc_tau_50_medium = tau_50_medium[:, 1] + 1j * tau_50_medium[:, 2]
y_axis_tau_50_medium = np.abs(dipole_acc_tau_50_medium)**2

omega_eV_tau_50_slow = tau_50_slow[:,0]*au_eV
dipole_acc_tau_50_slow = tau_50_slow[:, 1] + 1j * tau_50_slow[:, 2]
y_axis_tau_50_slow = np.abs(dipole_acc_tau_50_slow)**2

omega_eV_tau_50_extreme = tau_50_extreme[:,0]*au_eV
dipole_acc_tau_50_extreme = tau_50_extreme[:, 1] + 1j * tau_50_extreme[:, 2]
y_axis_tau_50_extreme = np.abs(dipole_acc_tau_50_extreme)**2

omega_eV_tau_10_fast = tau_10_fast[:,0]*au_eV
dipole_acc_tau_10_fast = tau_10_fast[:, 1] + 1j * tau_10_fast[:, 2]
y_axis_tau_10_fast = np.abs(dipole_acc_tau_10_fast)**2

omega_eV_tau_10_medium = tau_10_medium[:,0]*au_eV
dipole_acc_tau_10_medium = tau_10_medium[:, 1] + 1j * tau_10_medium[:, 2]
y_axis_tau_10_medium = np.abs(dipole_acc_tau_10_medium)**2

omega_eV_tau_10_slow = tau_10_slow[:,0]*au_eV
dipole_acc_tau_10_slow = tau_10_slow[:, 1] + 1j * tau_10_slow[:, 2]
y_axis_tau_10_slow = np.abs(dipole_acc_tau_10_slow)**2

omega_eV_tau_10_extreme = tau_10_extreme[:,0]*au_eV
dipole_acc_tau_10_extreme = tau_10_extreme[:, 1] + 1j * tau_10_extreme[:, 2]
y_axis_tau_10_extreme = np.abs(dipole_acc_tau_10_extreme)**2



plt.rc('text', usetex=True)

datasets = [
    [
        ("atol=1e-10, rtol=1e-8",  omega_eV_tau_50_fast,    y_axis_tau_50_fast),
        ("atol=1e-12, rtol=1e-10", omega_eV_tau_50_medium,  y_axis_tau_50_medium),
        ("atol=1e-14, rtol=1e-12", omega_eV_tau_50_slow,    y_axis_tau_50_slow),
        ("atol=1e-16, rtol=1e-14", omega_eV_tau_50_extreme, y_axis_tau_50_extreme),
    ],
    [
        ("atol=1e-10, rtol=1e-8",  omega_eV_tau_10_fast,    y_axis_tau_10_fast),
        ("atol=1e-12, rtol=1e-10", omega_eV_tau_10_medium,  y_axis_tau_10_medium),
        ("atol=1e-14, rtol=1e-12", omega_eV_tau_10_slow,    y_axis_tau_10_slow),
        ("atol=1e-16, rtol=1e-14", omega_eV_tau_10_extreme, y_axis_tau_10_extreme),
    ],
]

row_titles = [r"$\tau = 50$ meV", r"$\tau = 10$ meV"]

fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=False)

for row in range(2):
    for col in range(4):
        label, omega_eV, y_axis = datasets[row][col]

        # normalize like the reference script
        index = np.argmax(y_axis)
        omega_0 = omega_eV[index]
        x_val = omega_eV / omega_0
        y_val = y_axis / y_axis[index]

        ax = axes[row, col]
        ax.plot(x_val, y_val, linewidth=2)
        ax.set_yscale("log")
        ax.set_xlim(0, 11)
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 11, 1)))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.grid(which='major', linestyle='-', linewidth=0.8)
        ax.set_title(f"{row_titles[row]}, {label}", fontsize=11)

        if row == 1:
            ax.set_xlabel(r'$\omega /\omega_0$', fontsize=16)
        if col == 0:
            ax.set_ylabel(r'$|\ddot{p}(\omega)|^2 / |\ddot{p}(\omega_0)|^2$', fontsize=14)

        ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()



#===========================================INTENSITY=============
#tol kept at a_tol = 1e-14, r_tol=1e-12
tau_50_I12 = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_1fd5cc3de1fb0457/dipole_acc.txt")
tau_50_I13 = tau_50_slow
tau_50_I14 = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_8c971e444cb61b8d/dipole_acc.txt")
tau_50_I15 = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_ecd56d4d59a5834/dipole_acc.txt")

tau_10_I12 = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_4a9b902e103e69b2/dipole_acc.txt")
tau_10_I13 = tau_10_slow
tau_10_I14 = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_75fa9b03f0de2ac8/dipole_acc.txt")
tau_10_I15 = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_triangle_d9b0b03da2239cb1/dipole_acc.txt")

omega_eV_tau_50_I12 = tau_50_I12[:,0]*au_eV
dipole_acc_tau_50_I12 = tau_50_I12[:, 1] + 1j * tau_50_I12[:, 2]
y_axis_tau_50_I12 = np.abs(dipole_acc_tau_50_I12)**2

omega_eV_tau_50_I13 = tau_50_I13[:,0]*au_eV
dipole_acc_tau_50_I13 = tau_50_I13[:, 1] + 1j * tau_50_I13[:, 2]
y_axis_tau_50_I13 = np.abs(dipole_acc_tau_50_I13)**2

omega_eV_tau_50_I14 = tau_50_I14[:,0]*au_eV
dipole_acc_tau_50_I14 = tau_50_I14[:, 1] + 1j * tau_50_I14[:, 2]
y_axis_tau_50_I14 = np.abs(dipole_acc_tau_50_I14)**2

omega_eV_tau_50_I15 = tau_50_I15[:,0]*au_eV
dipole_acc_tau_50_I15 = tau_50_I15[:, 1] + 1j * tau_50_I15[:, 2]
y_axis_tau_50_I15 = np.abs(dipole_acc_tau_50_I15)**2

omega_eV_tau_10_I12 = tau_10_I12[:,0]*au_eV
dipole_acc_tau_10_I12 = tau_10_I12[:, 1] + 1j * tau_10_I12[:, 2]
y_axis_tau_10_I12 = np.abs(dipole_acc_tau_10_I12)**2

omega_eV_tau_10_I13 = tau_10_I13[:,0]*au_eV
dipole_acc_tau_10_I13 = tau_10_I13[:, 1] + 1j * tau_10_I13[:, 2]
y_axis_tau_10_I13 = np.abs(dipole_acc_tau_10_I13)**2

omega_eV_tau_10_I14 = tau_10_I14[:,0]*au_eV
dipole_acc_tau_10_I14 = tau_10_I14[:, 1] + 1j * tau_10_I14[:, 2]
y_axis_tau_10_I14 = np.abs(dipole_acc_tau_10_I14)**2

omega_eV_tau_10_I15 = tau_10_I15[:,0]*au_eV
dipole_acc_tau_10_I15 = tau_10_I15[:, 1] + 1j * tau_10_I15[:, 2]
y_axis_tau_10_I15 = np.abs(dipole_acc_tau_10_I15)**2

datasets_I = [
    [
        (r"$I=10^{12}$", omega_eV_tau_50_I12, y_axis_tau_50_I12),
        (r"$I=10^{13}$", omega_eV_tau_50_I13, y_axis_tau_50_I13),
        (r"$I=10^{14}$", omega_eV_tau_50_I14, y_axis_tau_50_I14),
        (r"$I=10^{15}$", omega_eV_tau_50_I15, y_axis_tau_50_I15),
    ],
    [
        (r"$I=10^{12}$", omega_eV_tau_10_I12, y_axis_tau_10_I12),
        (r"$I=10^{13}$", omega_eV_tau_10_I13, y_axis_tau_10_I13),
        (r"$I=10^{14}$", omega_eV_tau_10_I14, y_axis_tau_10_I14),
        (r"$I=10^{15}$", omega_eV_tau_10_I15, y_axis_tau_10_I15),
    ],
]

row_titles = [r"$\tau = 50meV$ fs", r"$\tau = 10$ meV"]

fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=False)

for row in range(2):
    for col in range(4):
        label, omega_eV, y_axis = datasets_I[row][col]

        index = np.argmax(y_axis)
        omega_0 = omega_eV[index]
        x_val = omega_eV / omega_0
        y_val = y_axis / y_axis[index]

        ax = axes[row, col]
        ax.plot(x_val, y_val, linewidth=2)
        ax.set_yscale("log")
        ax.set_xlim(0, 11)
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 11, 1)))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.grid(which='major', linestyle='-', linewidth=0.8)
        ax.set_title(f"{row_titles[row]}, {label}", fontsize=11)

        if row == 1:
            ax.set_xlabel(r'$\omega /\omega_0$', fontsize=16)
        if col == 0:
            ax.set_ylabel(r'$|\ddot{p}(\omega)|^2 / |\ddot{p}(\omega_0)|^2$', fontsize=14)

        ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
