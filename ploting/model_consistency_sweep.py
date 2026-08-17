"""
SI-style doping sweep of the induced-current spectrum |J(w)|.

For every doping mu and every implementation (L0=Hartree, L1=Zeeman, L2=Peierls)
we reuse the *identical* pipeline behind panel (d):
    load current_time_evolution.txt  ->  interpolate onto a common uniform time grid
    ->  rFFT  ->  |J(w)| = sqrt(|Jx|^2+|Jy|^2)
Stacking over mu gives a 2D map |J|(hw, mu): the plasmon dispersing with doping.
The Peierls-minus-Hartree difference map exposes the (tiny) residual.

Run:  python3 model_consistency_sweep.py            # compute + save + plot
      python3 model_consistency_sweep.py --plot     # replot from cached .npz
"""
import os, sys, glob, re
import numpy as np

ROOT="/home/soeren/University/masters/2.semester/ISA/data_LLM"
GEOMS={
 "AC bowtie":"sweep_data_mu_armchair_bowtie_10x10_rot90",
 "AC triangle":"sweep_data_mu_armchair_triangle_14x14_rot90",
 "ZZ bowtie":"sweep_data_mu_zigzag_bowtie_15x15_rot0",
 "ZZ triangle":"sweep_data_mu_zigzag_triangle_22x22_rot0",
}
HA=27.2114          # a.u. -> eV
NT=4000             # points on the common uniform time grid
EMAX=4.0            # crop maps to hw <= EMAX (eV)
_here=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(_here,"model_consistency_sweep.npz")

def load_JxJy(path):
    d=np.loadtxt(path)
    return d[:,0], d[:,1:3]          # t, [Jx,Jy]

def mu_list(base):
    mus=set()
    for p in glob.glob(os.path.join(base,"L2_mu_*")):
        m=re.search(r"L2_mu_([0-9.]+)$",p)
        if m: mus.add(m.group(1))
    return sorted(mus,key=float)

def compute():
    data={}
    for name,g in GEOMS.items():
        base=os.path.join(ROOT,g)
        mus=mu_list(base)
        # ---- pass 1: load everything, find common final time ----
        raw={}   # (mu,L) -> (t,Y)
        tmax=np.inf
        for mu in mus:
            for L in ["L0","L1","L2"]:
                fp=os.path.join(base,f"{L}_mu_{mu}","current_time_evolution.txt")
                if not os.path.exists(fp):
                    raw[(mu,L)]=None; continue
                t,Y=load_JxJy(fp)
                raw[(mu,L)]=(t,Y)
                tmax=min(tmax,t[-1])
        tg=np.linspace(0.0,tmax,NT)
        dt=tg[1]-tg[0]
        w=2*np.pi*np.fft.rfftfreq(NT,d=dt)      # angular freq (a.u.)
        E=w*HA                                  # photon energy (eV)
        keep=E<=EMAX
        E=E[keep]
        # ---- pass 2: interpolate + FFT -> |J|(hw) per (mu,L) ----
        maps={L:np.full((len(mus),keep.sum()),np.nan) for L in ["L0","L1","L2"]}
        for i,mu in enumerate(mus):
            for L in ["L0","L1","L2"]:
                r=raw[(mu,L)]
                if r is None: continue
                t,Y=r
                Yg=np.column_stack([np.interp(tg,t,Y[:,k]) for k in range(2)])
                F=np.fft.rfft(Yg,axis=0)
                mag=np.sqrt(np.abs(F[:,0])**2+np.abs(F[:,1])**2)[keep]
                maps[L][i,:]=mag
        data[name]=dict(mu=np.array([float(m) for m in mus]),E=E,
                        L0=maps["L0"],L1=maps["L1"],L2=maps["L2"])
        print(f"{name:12s}  mu:{len(mus)}  hw-bins:{len(E)}  peak|J|(L0)={np.nanmax(maps['L0']):.4g}")
    np.savez(CACHE,**{f"{n}__{k}":v for n,d in data.items() for k,v in d.items()},
             names=np.array(list(data.keys())))
    print("cached ->",CACHE)
    return data

def load_cache():
    z=np.load(CACHE,allow_pickle=True)
    names=[str(x) for x in z["names"]]
    return {n:{k:z[f"{n}__{k}"] for k in ("mu","E","L0","L1","L2")} for n in names}

def plot(data):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.size":10.5})
    names=list(data.keys())
    DIFF=0.005          # +/- 0.5 % of peak, shared diverging scale
    fig=plt.figure(figsize=(12.2,11.0))
    gs=GridSpec(len(names),5,width_ratios=[1,1,1,1,0.06],
                hspace=0.28,wspace=0.12,left=0.09,right=0.93,top=0.93,bottom=0.06)
    coltitles=["Hartree (L0)","Zeeman (L1)","Peierls (L2)","Peierls $-$ Hartree"]
    for r,name in enumerate(names):
        d=data[name]; mu=d["mu"]; E=d["E"]
        gmax=np.nanmax(d["L0"])
        ext=[E.min(),E.max(),mu.min(),mu.max()]
        ims=[]
        for c,L in enumerate(["L0","L1","L2"]):
            ax=fig.add_subplot(gs[r,c])
            im=ax.imshow(d[L]/gmax,origin="lower",aspect="auto",extent=ext,
                         cmap="magma",vmin=0,vmax=1)
            ims.append(im)
            if r==0: ax.set_title(coltitles[c],fontsize=11)
            if c==0: ax.set_ylabel(f"{name}\n$\\mu$ (eV)")
            else: ax.set_yticklabels([])
            if r==len(names)-1: ax.set_xlabel("$\\hbar\\omega$ (eV)")
            else: ax.set_xticklabels([])
        # difference column
        axd=fig.add_subplot(gs[r,3])
        D=(d["L2"]-d["L0"])/gmax
        imd=axd.imshow(D,origin="lower",aspect="auto",extent=ext,
                       cmap="RdBu_r",vmin=-DIFF,vmax=DIFF)
        if r==0: axd.set_title(coltitles[3],fontsize=11)
        axd.set_yticklabels([])
        if r==len(names)-1: axd.set_xlabel("$\\hbar\\omega$ (eV)")
        else: axd.set_xticklabels([])
        mx=np.nanmax(np.abs(D))*100
        axd.text(0.96,0.94,f"max$|\\Delta|$={mx:.2f}%",transform=axd.transAxes,
                 ha="right",va="top",fontsize=8.5,color="k",
                 bbox=dict(boxstyle="round",fc="white",ec="0.6",alpha=0.85))
        # per-row colorbars
        cax=fig.add_subplot(gs[r,4])
        fig.colorbar(ims[-1],cax=cax)
        if r==0: cax.set_title("$|J|/|J|_{\\max}$",fontsize=8.5,pad=6)
    # one shared difference colorbar along the bottom of column 4
    cbax=fig.add_axes([0.712,0.028,0.14,0.012])
    cb=fig.colorbar(imd,cax=cbax,orientation="horizontal",ticks=[-DIFF,0,DIFF])
    cb.set_ticklabels(["$-0.5\\%$","0","$+0.5\\%$"])
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("Doping sweep of the induced-current spectrum $|J(\\omega)|$:  "
                 "plasmon dispersion is identical across the three implementations",
                 fontsize=13,y=0.965)
    out="/home/soeren/University/masters/2.semester/ISA/scr/model_consistency_sweep_dispersion.pdf"
    fig.savefig(out,bbox_inches="tight")
    fig.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
    print("saved",out)

if __name__=="__main__":
    if "--plot" in sys.argv and os.path.exists(CACHE):
        plot(load_cache())
    else:
        plot(compute())
