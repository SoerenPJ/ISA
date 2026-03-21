import numpy as np
import matplotlib.pyplot as plt

# my Hamiltonian
my_HTB = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_afdd026a1ed5f824/HTB.txt")
my_HTBNxN = my_HTB[:,0::2] + 1j*my_HTB[:,1::2]

my_v_ee = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_afdd026a1ed5f824/V_ee.txt")

my_points = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_afdd026a1ed5f824/lattice_points.txt")
print(my_points.shape)

# line Hamiltonian
line_HTB = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/Pentalene_0-dir_5.0cells_dir0_H0k.dat")
line_HTB = line_HTB[1:]  # remove header
line_HTB_complex = line_HTB[0::2] + 1j*line_HTB[1::2]
line_HTBNxN = line_HTB_complex.reshape(46,46)

line_V_ee = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/Pentalene_0-dir_5.0cells_coulomb-graphene.dat")
#line_V_ee = line_V_ee[1:]  # remove header
line_V_ee_complex = line_V_ee[0::2] + 1j*line_V_ee[1::2]
line_V_eeNxN = line_V_ee.reshape(46,46)


line_points = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/Pentalene_0-dir_5.0cells_r.dat",skiprows=1,usecols=(0,1))
print(line_points.shape)
#===========================HTB=======================#
diff = my_HTBNxN - line_HTBNxN
abs_diff = np.abs(diff)

print("max difference:", np.max(np.abs(diff)))

# Basic Hermiticity checks
print("Hermiticity my_HTBNxN:", np.max(np.abs(my_HTBNxN - my_HTBNxN.conj().T)))
print("Hermiticity line_HTBNxN:", np.max(np.abs(line_HTBNxN - line_HTBNxN.conj().T)))

# Compare eigenvalue spectra (sorted)
evals_my, _ = np.linalg.eigh(my_HTBNxN)
evals_line, _ = np.linalg.eigh(line_HTBNxN)
my_eig = np.loadtxt("/home/soeren/University/masters/2.semester/ISA/scr/Simulations/graphene_zigzag_afdd026a1ed5f824/eigenvalues.txt")
my_eig = my_eig[:,0::2] + 1j*[]
evals_diff = np.sort(my_eig) - np.sort(evals_line)
print("max abs diff eigenvalues (HTB):", evals_diff)


plt.figure(figsize=(6,6))
plt.imshow(abs_diff, origin="lower")
plt.colorbar(label="|ΔH|")
plt.title("Difference between Hamiltonians")
plt.xlabel("j")
plt.ylabel("i")



plt.figure(figsize=(6,6))
plt.imshow(np.real(line_HTBNxN), origin="lower")
plt.colorbar(label="|ΔH|")
plt.title("Line Hamiltonians")
plt.xlabel("j")
plt.ylabel("i")


plt.figure(figsize=(6,6))
plt.imshow(np.real(my_HTBNxN), origin="lower")
plt.colorbar(label="|ΔH|")
plt.title("my  Hamiltonians")
plt.xlabel("j")
plt.ylabel("i")
plt.show()


#===============================V_ee====================#
diff_vee = np.abs(my_v_ee - line_V_eeNxN) 
plt.figure(figsize=(6,6))
plt.imshow(np.real(diff_vee), origin="lower")
plt.colorbar(label="|ΔH|")
plt.title("Difference between Interaction matrix")
plt.xlabel("j")
plt.ylabel("i")


plt.figure(figsize=(6,6))
plt.imshow(np.real(my_v_ee), origin="lower")
plt.colorbar(label="|ΔH|")
plt.title("me")
plt.xlabel("j")
plt.ylabel("i")


plt.figure(figsize=(6,6))
plt.imshow(np.real(line_V_eeNxN), origin="lower")
plt.colorbar(label="|ΔH|")
plt.title("Line")
plt.xlabel("j")
plt.ylabel("i")
plt.show()

#======================points==============
plt.figure(figsize=(6,6))

plt.scatter(my_points[:,0], my_points[:,1], s=60, label="mine")
plt.scatter(line_points[:,0], line_points[:,1], s=20, label="teacher")

plt.legend()
plt.gca().set_aspect("equal")
plt.title("Lattice comparison")

plt.show()

from scipy.spatial.distance import cdist

dist_matrix = cdist(my_points, line_points)

print("max closest distance:", np.max(np.min(dist_matrix, axis=1)))

# Try to infer a site-index permutation from nearest neighbors
nearest_idx = np.argmin(dist_matrix, axis=1)  # for each of my points, closest teacher point

if np.unique(nearest_idx).size == nearest_idx.size:
    print("All nearest neighbors are unique – using as permutation.")
else:
    print("Warning: some nearest neighbors are reused; permutation is approximate.")

perm = nearest_idx

# Reorder my matrices into teacher index order
H_perm = my_HTBNxN[np.ix_(perm, perm)]
V_perm = my_v_ee[np.ix_(perm, perm)]

print("max abs diff HTB after permutation:", np.max(np.abs(H_perm - line_HTBNxN)))
print("max abs diff V_ee after permutation:", np.max(np.abs(V_perm - line_V_eeNxN)))
