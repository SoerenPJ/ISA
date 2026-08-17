import numpy as np, os
'''
ROOT="/home/soeren/University/masters/2.semester/ISA/scr/data_LLM"
MU="3.52"
GEOMS={
 "AC bowtie":"sweep_data_mu_armchair_bowtie_2x2_rot90",
 "AC triangle":"sweep_data_mu_armchair_triangle_2x2_rot90",
 "ZZ bowtie":"sweep_data_mu_zigzag_bowtie_2x2_rot0",
 "ZZ triangle":"sweep_data_mu_zigzag_triangle_2x2_rot0",
}
'''



ROOT="/home/soeren/University/masters/2.semester/ISA/data_LLM"
MU="3.52"
GEOMS={
 "AC bowtie":"sweep_data_mu_armchair_bowtie_10x10_rot90",
 "AC triangle":"sweep_data_mu_armchair_triangle_14x14_rot90",
 "ZZ bowtie":"sweep_data_mu_zigzag_bowtie_15x15_rot0",
 "ZZ triangle":"sweep_data_mu_zigzag_triangle_22x22_rot0",
}

def load_t(path):
    d=np.loadtxt(path)
    return d[:,0], d[:,1:]

def bz_biotsavart_path(folder):
    # The Biot-Savart induced field file is named inconsistently across sweeps:
    #   small runs -> B_ind_z_sc_time_evolution.txt  (plain B_ind_z is the curl field)
    #   large runs -> B_ind_z_time_evolution.txt      (curl field is B_ind_z_curl)
    # Pick by header: Biot-Savart columns are "B_z_*", curl columns are "B_z_curl_*".
    for fn in ("B_ind_z_sc_time_evolution.txt","B_ind_z_time_evolution.txt"):
        p=os.path.join(folder,fn)
        if os.path.exists(p):
            with open(p) as f: hdr=f.readline()
            if "curl" not in hdr:
                return p
    raise FileNotFoundError(f"no Biot-Savart B_z file found in {folder}")

def to_grid(t, Y, tg):
    # interpolate each column onto common uniform grid tg
    if Y.ndim==1: Y=Y[:,None]
    out=np.empty((len(tg),Y.shape[1]))
    for k in range(Y.shape[1]):
        out[:,k]=np.interp(tg,t,Y[:,k])
    return out

def spectrum(tg, Y):
    # Y shape (Nt, Ncomp); return freqs (angular) and complex FFT per comp
    dt=tg[1]-tg[0]
    F=np.fft.rfft(Y,axis=0)
    f=np.fft.rfftfreq(len(tg),d=dt)      # ordinary freq (cycles per a.u. time)
    w=2*np.pi*f                          # angular freq (a.u.)
    return w,F

def relerr(a,b):
    # relative L2 norm of complex vectors
    return np.linalg.norm(a-b)/np.linalg.norm(b)

results={}
for name,g in GEOMS.items():
    base=os.path.join(ROOT,g)
    # ---- common uniform time grid from L2 net current ----
    tc={};Jc={}
    for L in ["L0","L1","L2"]:
        t,Y=load_t(os.path.join(base,f"{L}_mu_{MU}","current_time_evolution.txt"))
        tc[L]=t;Jc[L]=Y
    tmax=min(tc[L][-1] for L in tc)
    Nt=4000
    tg=np.linspace(0,tmax,Nt)
    # net current magnitude |J|=sqrt(Jx^2+Jy^2) spectrum, per model
    netspec={}
    for L in ["L0","L1","L2"]:
        Yg=to_grid(tc[L],Jc[L],tg)
        w,F=spectrum(tg,Yg)             # F: (Nf,2) Jx,Jy
        Fmag=np.sqrt(np.abs(F[:,0])**2+np.abs(F[:,1])**2)
        netspec[L]=(w,F,Fmag)
    w=netspec["L2"][0]
    # resonance = peak of net-current magnitude spectrum (L2), ignore w=0
    mask=w>0.02
    ires=np.where(mask)[0][np.argmax(netspec["L2"][2][mask])]
    wres=w[ires]

    # net current relative errors: peak-normalized max deviation of |J|(w) (mirrors extinction)
    mm=w>0.02
    m0=netspec["L0"][2];m1=netspec["L1"][2];m2=netspec["L2"][2]
    pk=np.max(m0[mm])
    e_net_10=np.max(np.abs(m1[mm]-m0[mm]))/pk
    e_net_20=np.max(np.abs(m2[mm]-m0[mm]))/pk
    e_net_21=np.max(np.abs(m2[mm]-m1[mm]))/np.max(m1[mm])

    # ---- bond currents L1 vs L2 ----
    bonds={}
    for L in ["L1","L2"]:
        t,Y=load_t(os.path.join(base,f"{L}_mu_{MU}","J_bond_time_evolution.txt"))
        Yg=to_grid(t,Y,tg)
        _,F=spectrum(tg,Yg)
        bonds[L]=F[ires,:]              # complex amplitude per bond at resonance
    e_bond=relerr(bonds["L2"],bonds["L1"])

    # ---- induced Bz L1 vs L2 (Biot-Savart field in both, selected by header) ----
    bz={}
    for L in ["L1","L2"]:
        t,Y=load_t(bz_biotsavart_path(os.path.join(base,f"{L}_mu_{MU}")))
        Yg=to_grid(t,Y,tg)
        _,F=spectrum(tg,Yg)
        bz[L]=F[ires,:]
    e_bz=relerr(bz["L2"],bz["L1"])
    e_bz_max=np.max(np.abs(bz["L2"]-bz["L1"]))/np.max(np.abs(bz["L1"]))
    e_bond_max=np.max(np.abs(bonds["L2"]-bonds["L1"]))/np.max(np.abs(bonds["L1"]))

    # ---- extinction (sanity, three models) ----
    sig={}
    for L in ["L0","L1","L2"]:
        s=np.loadtxt(os.path.join(base,f"{L}_mu_{MU}","sigma_ext.txt"))
        sig[L]=s
    # interp L1,L2 onto L0 grid
    x0=sig["L0"][:,0]
    s0=sig["L0"][:,1]
    s1=np.interp(x0,sig["L1"][:,0],sig["L1"][:,1])
    s2=np.interp(x0,sig["L2"][:,0],sig["L2"][:,1])
    e_sig_10=np.max(np.abs(s1-s0))/np.max(np.abs(s0))
    e_sig_20=np.max(np.abs(s2-s0))/np.max(np.abs(s0))

    results[name]=dict(wres=wres,
        e_sig_10=e_sig_10,e_sig_20=e_sig_20,
        e_net_10=e_net_10,e_net_20=e_net_20,e_net_21=e_net_21,
        e_bond=e_bond,e_bz=e_bz,e_bond_max=e_bond_max,e_bz_max=e_bz_max,
        bonds=bonds,bz=bz,netspec=netspec,w=w,ires=ires)

print(f"{'geom':12s} {'wres':>6s} | {'sig20%':>7s} | {'net20%':>7s} | {'bondL2%':>8s}{'bondMx%':>8s} | {'BzL2%':>7s}{'BzMx%':>7s}")
for name,r in results.items():
    print(f"{name:12s} {r['wres']:6.3f} | {r['e_sig_20']*100:7.3f} | {r['e_net_20']*100:7.3f} | "
          f"{r['e_bond']*100:8.3f}{r['e_bond_max']*100:8.3f} | {r['e_bz']*100:7.3f}{r['e_bz_max']*100:7.3f}")

_here=os.path.dirname(os.path.abspath(__file__))
np.save(os.path.join(_here,"model_consistency_results.npy"),results,allow_pickle=True)
