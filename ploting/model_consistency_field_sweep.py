"""
Doping sweep of SPATIAL current / field observables at the plasmon resonance.

Unlike the net-current spectrum |J(w)| (which is just the polarizability alpha,
i.e. the extinction re-plotted), the quantities here are NOT contained in
sigma_ext, because they keep the spatial structure the net dipole throws away:

    Sigma|J_ll'(w_res)|      total resonant bond-current "activity"  (over all bonds)
    max_r |B_z^ind(w_res)|   peak self-induced magnetic field       (over all sites)

Both are followed vs doping mu and compared between the two implementations that
produce a magnetic field: Zeeman (L1) and Peierls (L2).  (Hartree/L0 has no field
by construction.)  This is the current-and-field investigation with content that
the linear optical response cannot see.

Method (per geometry, per mu):
  * find the plasmon frequency w_res(mu) from the net-current spectrum,
  * evaluate every bond / site at that single w_res via a direct trapezoidal DFT
    on the native (non-uniform) time grid  ->  X(w)=∫ Y(t) e^{-i w t} dt,
  * reduce to Sigma|J| (bonds) and max|B| (sites).

Run:  python3 model_consistency_field_sweep.py            # compute + cache + plot
      python3 model_consistency_field_sweep.py --plot     # replot from cache
"""
import os, sys, glob, re
import numpy as np
import pandas as pd

ROOT="/home/soeren/University/masters/2.semester/ISA/data_LLM"
GEOMS={
 "AC bowtie":"sweep_data_mu_armchair_bowtie_10x10_rot90",
 "AC triangle":"sweep_data_mu_armchair_triangle_14x14_rot90",
 "ZZ bowtie":"sweep_data_mu_zigzag_bowtie_15x15_rot0",
 "ZZ triangle":"sweep_data_mu_zigzag_triangle_22x22_rot0",
}
HA=27.2114
MU_PAPER=3.52                       # doping used for Fig.2 of the paper
_here=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(_here,"model_consistency_field_sweep.npz")

def read_fast(path):
    df=pd.read_csv(path,sep=r"\s+",comment="#",header=None,dtype=np.float64)
    a=df.to_numpy()
    return a[:,0], a[:,1:]

def bz_biotsavart_path(folder):
    # Biot-Savart field: named B_ind_z_sc (small runs) or B_ind_z (large runs);
    # the curl field (B_ind_z_curl / header 'B_z_curl') must be excluded.
    for fn in ("B_ind_z_sc_time_evolution.txt","B_ind_z_time_evolution.txt"):
        p=os.path.join(folder,fn)
        if os.path.exists(p):
            with open(p) as f: hdr=f.readline()
            if "curl" not in hdr:
                return p
    raise FileNotFoundError(f"no Biot-Savart B_z file in {folder}")

def dft_at(t,Y,w):
    # single-frequency Fourier component per column, trapezoid on native grid
    ph=np.exp(-1j*w*t)
    return np.trapz(Y*ph[:,None],t,axis=0)      # (Ncols,) complex

def find_wres(t,J):
    # peak of |J(w)| scanned over the plasmon window (a.u.)
    wg=np.linspace(0.02,0.30,600)
    mag=np.array([abs(dft_at(t,J,w)).sum() for w in wg])
    return wg[mag.argmax()]

def mu_list(base):
    mus={re.search(r"L2_mu_([0-9.]+)$",p).group(1)
         for p in glob.glob(os.path.join(base,"L2_mu_*"))}
    return sorted(mus,key=float)

def compute():
    data={}
    for name,g in GEOMS.items():
        base=os.path.join(ROOT,g); mus=mu_list(base)
        mu=np.array([float(m) for m in mus])
        sumJ={"L1":np.full(len(mus),np.nan),"L2":np.full(len(mus),np.nan)}
        maxB={"L1":np.full(len(mus),np.nan),"L2":np.full(len(mus),np.nan)}
        wres=np.full(len(mus),np.nan)
        for i,m in enumerate(mus):
            # resonance from the net current (small file)
            tc,Jc=read_fast(os.path.join(base,f"L2_mu_{m}","current_time_evolution.txt"))
            w=find_wres(tc,Jc[:,:2]); wres[i]=w
            for L in ["L1","L2"]:
                fol=os.path.join(base,f"{L}_mu_{m}")
                tb,Jb=read_fast(os.path.join(fol,"J_bond_time_evolution.txt"))
                sumJ[L][i]=np.abs(dft_at(tb,Jb,w)).sum()
                tB,B=read_fast(bz_biotsavart_path(fol))
                maxB[L][i]=np.abs(dft_at(tB,B,w)).max()
        data[name]=dict(mu=mu,wres=wres,
                        sumJ_L1=sumJ["L1"],sumJ_L2=sumJ["L2"],
                        maxB_L1=maxB["L1"],maxB_L2=maxB["L2"])
        # headline invariance numbers (where the signal is non-trivial)
        def reldiff(a,b):
            mask=b>0.1*np.nanmax(b)
            return np.nanmax(np.abs(a[mask]-b[mask])/b[mask])*100
        print(f"{name:12s}  maxdev  Sigma|J|: {reldiff(sumJ['L2'],sumJ['L1']):.3f}%   "
              f"max|B|: {reldiff(maxB['L2'],maxB['L1']):.3f}%")
    np.savez(CACHE,**{f"{n}__{k}":v for n,d in data.items() for k,v in d.items()},
             names=np.array(list(data.keys())))
    print("cached ->",CACHE)
    return data

def load_cache():
    z=np.load(CACHE,allow_pickle=True)
    names=[str(x) for x in z["names"]]
    keys=("mu","wres","sumJ_L1","sumJ_L2","maxB_L1","maxB_L2")
    return {n:{k:z[f"{n}__{k}"] for k in keys} for n in names}

def plot(data):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10.5})
    names=list(data.keys())
    fig,axes=plt.subplots(2,len(names),figsize=(13.5,6.6),sharex=True)
    rows=[("$\\sum_{\\ell\\ell'}\\,|J_{\\ell\\ell'}(\\omega_{\\rm res})|$   (a.u.)",
           "sumJ_L1","sumJ_L2"),
          ("$\\max_{\\ell}\\,|B_z^{\\rm ind}(\\omega_{\\rm res})|$   (a.u.)",
           "maxB_L1","maxB_L2")]
    for r,(ylab,k1,k2) in enumerate(rows):
        for c,name in enumerate(names):
            ax=axes[r,c]; d=data[name]; mu=d["mu"]
            ax.plot(mu,d[k1],"-",color="#1f77b4",lw=2.2,label="Zeeman (L1)")
            ax.plot(mu,d[k2],"--",color="#d62728",lw=1.6,label="Peierls (L2)")
            ax.axvline(MU_PAPER,color="0.5",ls=":",lw=1)
            ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
            if r==0: ax.set_title(name,fontsize=11)
            if r==1: ax.set_xlabel("$\\mu$ (eV)")
            if c==0: ax.set_ylabel(ylab,fontsize=10)
            # invariance annotation
            b=d[k1]; a=d[k2]; msk=b>0.1*np.nanmax(b)
            dev=np.nanmax(np.abs(a[msk]-b[msk])/b[msk])*100
            ax.text(0.05,0.92,f"max dev {dev:.2f}%",transform=ax.transAxes,
                    fontsize=8.5,va="top",
                    bbox=dict(boxstyle="round",fc="white",ec="0.6",alpha=0.85))
            if r==0 and c==0:
                ax.legend(fontsize=8.5,frameon=False,loc="lower right")
    axes[0,-1].text(MU_PAPER,axes[0,-1].get_ylim()[1]*0.5,"  $\\mu$ of Fig.2",
                    color="0.4",fontsize=8,rotation=90,va="center")
    fig.suptitle("Doping dependence of the resonant bond current and self-induced magnetic field "
                 "(spatial observables not contained in $\\sigma_{\\rm ext}$)",fontsize=12.5,y=0.98)
    fig.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    out="/home/soeren/University/masters/2.semester/ISA/scr/model_consistency_field_sweep.pdf"
    fig.savefig(out,bbox_inches="tight"); fig.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
    print("saved",out)

if __name__=="__main__":
    if "--plot" in sys.argv and os.path.exists(CACHE):
        plot(load_cache())
    else:
        plot(compute())
