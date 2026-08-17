import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_here=os.path.dirname(os.path.abspath(__file__))
R=np.load(os.path.join(_here,"model_consistency_results.npy"),allow_pickle=True).item()
HA=27.2114  # a.u. -> eV
geoms=list(R.keys())
cols={"AC bowtie":"#1f77b4","AC triangle":"#ff7f0e","ZZ bowtie":"#2ca02c","ZZ triangle":"#d62728"}
mk  ={"AC bowtie":"o","AC triangle":"s","ZZ bowtie":"^","ZZ triangle":"D"}

plt.rcParams.update({"font.size":11,"axes.linewidth":0.8})
fig=plt.figure(figsize=(11,8.4))
gs=GridSpec(2,3,figure=fig,height_ratios=[1.0,1.05],hspace=0.42,wspace=0.34,
            left=0.08,right=0.985,top=0.90,bottom=0.09)

# ================= Panel A : summary bars =================
axA=fig.add_subplot(gs[0,:])
obs=[("$\\sigma_{\\rm ext}$","e_sig_20","#8c564b"),
     ("net current $J(\\omega)$","e_net_20","#9467bd"),
     ("bond current $J_{\\ell\\ell'}(\\omega_{\\rm res})$","e_bond","#17becf"),
     ("field $B_z^{\\rm ind}(\\omega_{\\rm res})$","e_bz","#e377c2")]
nG=len(geoms); nO=len(obs); w=0.8/nO
x=np.arange(nG)
for k,(lab,key,c) in enumerate(obs):
    vals=[R[g][key]*100 for g in geoms]
    axA.bar(x+(k-(nO-1)/2)*w,vals,w,label=lab,color=c,edgecolor="k",linewidth=0.4,zorder=3)
axA.axhline(0.5,ls="--",color="k",lw=1.1,zorder=2)
axA.text(nG-0.5,0.55,"0.5 % linear-response bound (paper)",ha="right",va="bottom",fontsize=9.5)
axA.set_yscale("log")
axA.set_ylim(5e-4,3)
axA.set_xticks(x); axA.set_xticklabels(geoms)
axA.set_ylabel("max. relative deviation\nbetween models (%)")
axA.set_title("(a)  Deviation of each observable across the three implementations "
              "(Hartree $\\to$ Zeeman $\\to$ Peierls)",fontsize=11,loc="left")
axA.legend(ncol=4,fontsize=9,loc="upper center",bbox_to_anchor=(0.5,1.20),frameon=False,
           columnspacing=1.3,handlelength=1.3)
axA.grid(axis="y",ls=":",alpha=0.5,zorder=0)

# ================= Panel B : bond-current parity =================
axB=fig.add_subplot(gs[1,0])
for g in geoms:
    a1=np.abs(R[g]["bonds"]["L1"]); a2=np.abs(R[g]["bonds"]["L2"])
    n=a1.max()
    axB.scatter(a1/n,a2/n,s=34,c=cols[g],marker=mk[g],edgecolor="k",linewidth=0.3,
                alpha=0.85,label=g,zorder=3)
axB.plot([0,1],[0,1],"k--",lw=1,zorder=2)
axB.set_xlabel("$|J_{\\ell\\ell'}|$  Zeeman (L1)  [norm.]")
axB.set_ylabel("$|J_{\\ell\\ell'}|$  Peierls (L2)  [norm.]")
axB.set_title("(b)  Bond currents at $\\omega_{\\rm res}$",fontsize=11,loc="left")
axB.set_xlim(-0.03,1.03); axB.set_ylim(-0.03,1.03); axB.set_aspect("equal")
mb=max(R[g]["e_bond"] for g in geoms)*100
axB.text(0.05,0.93,f"max dev. $\\leq$ {mb:.2f} %",transform=axB.transAxes,fontsize=10,
         bbox=dict(boxstyle="round",fc="white",ec="0.6"))

# ================= Panel C : Bz parity =================
axC=fig.add_subplot(gs[1,1])
for g in geoms:
    a1=np.abs(R[g]["bz"]["L1"]); a2=np.abs(R[g]["bz"]["L2"])
    n=a1.max()
    axC.scatter(a1/n,a2/n,s=34,c=cols[g],marker=mk[g],edgecolor="k",linewidth=0.3,
                alpha=0.85,zorder=3)
axC.plot([0,1],[0,1],"k--",lw=1,zorder=2)
axC.set_xlabel("$|B_z^{\\rm ind}|$  Zeeman (L1)  [norm.]")
axC.set_ylabel("$|B_z^{\\rm ind}|$  Peierls (L2)  [norm.]")
axC.set_title("(c)  Induced field at $\\omega_{\\rm res}$",fontsize=11,loc="left")
axC.set_xlim(-0.03,1.03); axC.set_ylim(-0.03,1.03); axC.set_aspect("equal")
mc=max(R[g]["e_bz"] for g in geoms)*100
axC.text(0.05,0.93,f"max dev. $\\leq$ {mc:.2f} %",transform=axC.transAxes,fontsize=10,
         bbox=dict(boxstyle="round",fc="white",ec="0.6"))

# ================= Panel D : net-current spectra overlay =================
axD=fig.add_subplot(gs[1,2])
g="ZZ triangle"
w=R[g]["w"]*HA
ls={"L0":"-","L1":"--","L2":":"}
lab={"L0":"Hartree (L0)","L1":"Zeeman (L1)","L2":"Peierls (L2)"}
lc ={"L0":"#333333","L1":"#1f77b4","L2":"#d62728"}
for L in ["L0","L1","L2"]:
    mag=R[g]["netspec"][L][2]
    axD.plot(w,mag/mag[w>0.5].max(),ls[L],color=lc[L],lw=1.8,label=lab[L])
axD.set_xlim(0,9)
axD.set_xlabel("photon energy $\\hbar\\omega$ (eV)")
axD.set_ylabel("$|J(\\omega)|$  (norm.)")
axD.set_title(f"(d)  Net current spectrum\n({g})",fontsize=11,loc="left")
axD.legend(fontsize=8.5,frameon=False,loc="upper right")

# shared geometry legend for B/C
handles=[plt.Line2D([],[],marker=mk[g],color="w",markerfacecolor=cols[g],
         markeredgecolor="k",markersize=8,label=g) for g in geoms]
fig.legend(handles=handles,ncol=4,fontsize=9.5,loc="lower center",
           bbox_to_anchor=(0.5,-0.002),frameon=False)

fig.suptitle("Self-induced current and magnetic field are invariant across the three model implementations",
             fontsize=13,y=0.975)
out="/home/soeren/University/masters/2.semester/ISA/scr/model_consistency_current_Bfield.pdf"
fig.savefig(out,bbox_inches="tight")
fig.savefig(out.replace(".pdf",".png"),dpi=160,bbox_inches="tight")
plt.show()
print("saved",out)
