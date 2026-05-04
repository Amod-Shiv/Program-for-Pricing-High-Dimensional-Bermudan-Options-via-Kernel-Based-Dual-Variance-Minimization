"""
Bermudan Max-Call Pricing: VM (dual variance minimization) & PO (pathwise optimization)
Ref: Li (2025) s10614-025-10933-0, Desai et al. (2012)
Supports d=2 and d=5 dimensions.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, os

OUT = os.path.dirname(os.path.abspath(__file__))
r_rate=0.05; delta_div=0.1; sig=0.2; T=3.0; N=9; dt=T/N; K_s=100
N_rep=30

def pr(*a,**kw): print(*a,**kw,flush=True)

def sim(s0,K,d,seed=None):
    if seed is not None: np.random.seed(seed)
    S=np.zeros((K,N+1,d)); S[:,0,:]=s0
    dr=(r_rate-delta_div-0.5*sig**2)*dt; vol=sig*np.sqrt(dt)
    for t in range(1,N+1):
        S[:,t,:]=S[:,t-1,:]*np.exp(dr+vol*np.random.randn(K,d))
    return S

def payoff(S,t):
    return np.maximum(np.max(S[:,t,:],axis=1)-K_s,0.0)

def disc(t): return np.exp(-r_rate*t*dt)

def basis(S_t,d):
    f=[np.ones(len(S_t))]
    for i in range(d): f.append(S_t[:,i])
    for i in range(d): f.append(S_t[:,i]**2)
    for i in range(d):
        for j in range(i+1,d): f.append(S_t[:,i]*S_t[:,j])
    return np.column_stack(f)

# ── LSMC with stopping rule ──
def lsmc(S,d):
    K=S.shape[0]
    cf=payoff(S,N)*disc(N); tau=np.full(K,N); betas={}
    for t in range(N-1,0,-1):
        ex=payoff(S,t)
        itm=ex>0
        if itm.sum()<max(10,d*3): continue
        X=basis(S[itm,t,:],d)
        Y=cf[itm]*np.exp(-r_rate*(tau[itm]-t)*dt)/disc(t)
        beta=np.linalg.lstsq(X,Y,rcond=None)[0]; betas[t]=beta
        cont=X@beta
        do_ex=ex[itm]>cont
        idx=np.where(itm)[0][do_ex]
        cf[idx]=ex[idx]*disc(t); tau[idx]=t
    prices=np.zeros(K)
    for k in range(K):
        prices[k]=payoff(S[k:k+1],int(tau[k])).item()*disc(int(tau[k]))
    return prices.mean(), prices.std()/np.sqrt(K), betas, tau

# ── VM: Andersen-Broadie dual upper bound ──
def vm_upper(S_ub,betas,d,n_sub=1500):
    K=S_ub.shape[0]; M=np.zeros((K,N+1))
    dr_=(r_rate-delta_div-0.5*sig**2)*dt; vol_=sig*np.sqrt(dt)
    for t in range(1,N+1):
        h_t=payoff(S_ub,t)
        if t==N:
            Q_t=h_t
        elif t in betas:
            C_t=basis(S_ub[:,t,:],d)@betas[t]
            Q_t=np.maximum(h_t,C_t)
        else:
            Q_t=h_t
        # E[Q_t|F_{t-1}] via sub-simulation
        E_Qt=np.zeros(K)
        for _ in range(n_sub):
            Z=np.random.randn(K,d)
            S_sub=S_ub[:,t-1,:]*np.exp(dr_+vol_*Z)
            h_sub=np.maximum(np.max(S_sub,axis=1)-K_s,0.0)
            if t==N:
                Q_sub=h_sub
            elif t in betas:
                C_sub=basis(S_sub,d)@betas[t]
                Q_sub=np.maximum(h_sub,C_sub)
            else:
                Q_sub=h_sub
            E_Qt+=Q_sub
        E_Qt/=n_sub
        M[:,t]=M[:,t-1]+disc(t)*(Q_t-E_Qt)
    vals=np.column_stack([disc(t)*payoff(S_ub,t)-M[:,t] for t in range(N+1)])
    return vals.max(axis=1)

# ── PO: Pathwise Optimization ──
def po_basis(S_t,d):
    f=[np.ones(len(S_t))]
    for i in range(d): f.append(S_t[:,i])
    for i in range(d): f.append(S_t[:,i]**2)
    for i in range(d):
        for j in range(i+1,d): f.append(S_t[:,i]*S_t[:,j])
    return np.column_stack(f)

def po_Ebasis(Sp,d):
    e1=np.exp((r_rate-delta_div)*dt)
    e2=np.exp((2*(r_rate-delta_div)+sig**2)*dt)
    e12=np.exp(2*(r_rate-delta_div)*dt)
    f=[np.ones(len(Sp))]
    for i in range(d): f.append(Sp[:,i]*e1)
    for i in range(d): f.append(Sp[:,i]**2*e2)
    for i in range(d):
        for j in range(i+1,d): f.append(Sp[:,i]*Sp[:,j]*e12)
    return np.column_stack(f)

def po_obj(rv,S,gd,d):
    K=S.shape[0]; pen=np.zeros((K,N+1)); bd=np.exp(-r_rate*dt)
    for p in range(1,N+1):
        phi=po_basis(S[:,p,:],d); Ep=po_Ebasis(S[:,p-1,:],d)
        dp=(phi-Ep)@rv
        for s in range(p,N+1): pen[:,s]+=bd**p*dp
    return (gd-pen).max(axis=1)

def run_experiment(s0,d,K_tr,K_ub,n_sub,seed):
    np.random.seed(seed)
    S_tr=sim(s0,K_tr,d)
    lb,lb_se,betas,tau=lsmc(S_tr,d)
    # VM upper
    S_ub=sim(s0,K_ub,d)
    V_vm=vm_upper(S_ub,betas,d,n_sub=n_sub)
    vm_hat=V_vm.mean(); vm_se=V_vm.std()/np.sqrt(K_ub)
    # PO upper
    gd_tr=np.column_stack([disc(t)*payoff(S_tr,t) for t in range(N+1)])
    nb=1+d+d+d*(d-1)//2
    def obj_po(rv): return po_obj(rv,S_tr[:min(2000,K_tr)],gd_tr[:min(2000,K_tr)],d).mean()
    res=minimize(obj_po,np.zeros(nb),method='Nelder-Mead',
                 options={'maxiter':5000,'xatol':1e-9,'fatol':1e-9})
    gd_ub=np.column_stack([disc(t)*payoff(S_ub,t) for t in range(N+1)])
    V_po=po_obj(res.x,S_ub,gd_ub,d)
    po_hat_tr=po_obj(res.x,S_tr,gd_tr,d).mean()
    po_hat=V_po.mean(); po_se=V_po.std()/np.sqrt(K_ub)
    return lb,vm_hat,vm_se,po_hat_tr,po_hat,po_se

if __name__=='__main__':
    t_total=time.time()
    refs2={90:(8.05,8.15),100:(13.88,14.01),110:(21.36,21.51)}
    refs5={90:(16.59,16.77),100:(26.13,26.34),110:(36.73,37.04)}
    s0_list=[90,100,110]
    R={}

    for dd,refs,K_tr,K_ub,nsub in [(2,refs2,4000,5000,2000),(5,refs5,4000,5000,2000)]:
        pr(f"\n{'='*60}\n  d={dd} Experiment ({N_rep} reps, K_tr={K_tr}, K_ub={K_ub})\n{'='*60}")
        t0=time.time()
        for s0 in s0_list:
            vm_ubs=[]; po_ubs=[]; po_hats=[]; lbs=[]
            for rep in range(N_rep):
                sd=dd*10000+s0*100+rep*13
                lb,vm_u,vm_se,po_h,po_u,po_se=run_experiment(s0,dd,K_tr,K_ub,nsub,sd)
                lbs.append(lb); vm_ubs.append(vm_u); po_ubs.append(po_u); po_hats.append(po_h)
                if rep%5==0:
                    pr(f"  s0={s0} r={rep}: LB={lb:.4f} VM_UB={vm_u:.4f} PO_UB={po_u:.4f}")
            R[('VM',dd,s0)]={'LB':np.array(lbs),'Vt':np.array(vm_ubs)}
            R[('PO',dd,s0)]={'LB':np.array(lbs),'Vh':np.array(po_hats),'Vt':np.array(po_ubs)}
        el=time.time()-t0; pr(f"  d={dd} time: {el:.1f}s ({el/60:.1f}min)")

    total_time=time.time()-t_total
    pr(f"\n  TOTAL RUNTIME: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Results ──
    # V0_tilde = lower bound (LSMC), V0_cap = upper bound (dual)
    # CI encompasses both: [LB - 1.96*se_LB, UB + 1.96*se_UB]
    for dd,refs in [(2,refs2),(5,refs5)]:
        pr(f"\n{'='*140}")
        pr(f"  RESULTS d={dd}  |  V0_tilde = Lower Bound (LSMC)  |  V0_cap = Upper Bound (Dual)")
        pr(f"{'='*140}")
        pr(f"{'':>5} |          --- VM ---                                |          --- PO ---")
        pr(f"{'s0':>5} | {'V0_tilde':>8} {'Std':>6} | {'V0_cap':>8} {'Std':>6} | {'95% CI':>22} | {'V0_tilde':>8} {'Std':>6} | {'V0_cap':>8} {'Std':>6} | {'95% CI':>22} | {'Ref':>14}")
        pr('-'*140)
        for s0 in s0_list:
            if ('VM',dd,s0) not in R: continue
            vm=R[('VM',dd,s0)]; po=R[('PO',dd,s0)]; ref=refs[s0]
            # VM
            vm_lb_m, vm_lb_s = vm['LB'].mean(), vm['LB'].std()
            vm_ub_m, vm_ub_s = vm['Vt'].mean(), vm['Vt'].std()
            vm_ci_lo = vm_lb_m - 1.96*vm_lb_s/np.sqrt(N_rep)
            vm_ci_hi = vm_ub_m + 1.96*vm_ub_s/np.sqrt(N_rep)
            # PO
            po_lb_m, po_lb_s = po['LB'].mean(), po['LB'].std()
            po_ub_m, po_ub_s = po['Vt'].mean(), po['Vt'].std()
            po_ci_lo = po_lb_m - 1.96*po_lb_s/np.sqrt(N_rep)
            po_ci_hi = po_ub_m + 1.96*po_ub_s/np.sqrt(N_rep)
            pr(f"{s0:>5} | {vm_lb_m:8.2f} {vm_lb_s:6.2f} | {vm_ub_m:8.2f} {vm_ub_s:6.2f} | [{vm_ci_lo:8.2f},{vm_ci_hi:8.2f}] | {po_lb_m:8.2f} {po_lb_s:6.2f} | {po_ub_m:8.2f} {po_ub_s:6.2f} | [{po_ci_lo:8.2f},{po_ci_hi:8.2f}] | [{ref[0]},{ref[1]}]")

    # ── LaTeX Table ──
    tex=r"""\begin{table}[htbp]
\centering
\caption{Bermudan max-call pricing results ($K=100$, $T=3$, $N=9$, $r=0.05$, $\delta=0.1$, $\sigma=0.2$).
$\tilde{V}_0$: lower bound (LSMC), $\hat{V}_0$: upper bound (dual method).
VM: variance minimization, PO: pathwise optimization.
Standard deviations over """+str(N_rep)+r""" runs. 95\% CI encompasses both bounds.}
\label{tab:comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{cc|cccccc|cccccc}
\hline
& & \multicolumn{6}{c|}{VM (Variance Minimization)} & \multicolumn{6}{c}{PO (Pathwise Optimization)} \\
$d$ & $s_0$ & $\tilde{V}_0$ & Std. & $\hat{V}_0$ & Std. & \multicolumn{2}{c|}{95\% CI} & $\tilde{V}_0$ & Std. & $\hat{V}_0$ & Std. & \multicolumn{2}{c}{95\% CI} \\
\hline
"""
    for dd,refs in [(2,refs2),(5,refs5)]:
        for s0 in s0_list:
            if ('VM',dd,s0) not in R: continue
            vm=R[('VM',dd,s0)]; po=R[('PO',dd,s0)]; ref=refs[s0]
            vm_lb_m,vm_lb_s=vm['LB'].mean(),vm['LB'].std()
            vm_ub_m,vm_ub_s=vm['Vt'].mean(),vm['Vt'].std()
            vm_ci_lo=vm_lb_m-1.96*vm_lb_s/np.sqrt(N_rep)
            vm_ci_hi=vm_ub_m+1.96*vm_ub_s/np.sqrt(N_rep)
            po_lb_m,po_lb_s=po['LB'].mean(),po['LB'].std()
            po_ub_m,po_ub_s=po['Vt'].mean(),po['Vt'].std()
            po_ci_lo=po_lb_m-1.96*po_lb_s/np.sqrt(N_rep)
            po_ci_hi=po_ub_m+1.96*po_ub_s/np.sqrt(N_rep)
            tex+=f"{dd} & {s0} & {vm_lb_m:.2f} & {vm_lb_s:.2f} & {vm_ub_m:.2f} & {vm_ub_s:.2f} & [{vm_ci_lo:.2f}, & {vm_ci_hi:.2f}] & {po_lb_m:.2f} & {po_lb_s:.2f} & {po_ub_m:.2f} & {po_ub_s:.2f} & [{po_ci_lo:.2f}, & {po_ci_hi:.2f}] \\\\\n"
        tex+=r"\hline"+"\n"
    tex+=r"""\end{tabular}}
\end{table}"""
    with open(os.path.join(OUT,'comparison_table.tex'),'w') as f: f.write(tex)
    pr("\nLaTeX saved to comparison_table.tex")

    # ── Convergence: error% vs K ──
    K_vals=[500,1000,1500,2000,3000,4000]
    for dd,refs,title in [(2,refs2,'d=2'),(5,refs5,'d=5')]:
        ref_mid=np.mean(refs[100])
        fig,ax=plt.subplots(figsize=(9,6))
        for meth,col,mk in [('VM','#1565C0','o'),('PO','#E65100','s')]:
            errs=[]
            for Kv in K_vals:
                vals=[]
                for rep in range(3):
                    sd=dd*50000+100*100+rep*31+Kv
                    np.random.seed(sd)
                    S_tr=sim(100,Kv,dd)
                    if meth=='VM':
                        _,_,betas,_=lsmc(S_tr,dd)
                        S_te=sim(100,3000,dd)
                        vv=vm_upper(S_te,betas,dd,n_sub=1000)
                        vals.append(vv.mean())
                    else:
                        gd=np.column_stack([disc(t)*payoff(S_tr,t) for t in range(N+1)])
                        nb_=1+dd+dd+dd*(dd-1)//2
                        def obj_(rv): return po_obj(rv,S_tr[:min(1500,Kv)],gd[:min(1500,Kv)],dd).mean()
                        res_=minimize(obj_,np.zeros(nb_),method='Nelder-Mead',options={'maxiter':3000})
                        S_te=sim(100,3000,dd)
                        gd_te=np.column_stack([disc(t)*payoff(S_te,t) for t in range(N+1)])
                        vals.append(po_obj(res_.x,S_te,gd_te,dd).mean())
                err=abs(np.mean(vals)-ref_mid)/ref_mid*100
                errs.append(err)
                pr(f"  Conv {title} {meth} K={Kv}: mean={np.mean(vals):.3f} err={err:.2f}%")
            ax.plot(K_vals,errs,'-'+mk,color=col,label=f'{meth}',linewidth=2,markersize=8)
        ax.set_xlabel('Number of Training Samples (K)',fontsize=13)
        ax.set_ylabel('Error % (relative to reference)',fontsize=13)
        ax.set_title(f'Pricing Error vs Sample Size ({title})',fontsize=15)
        ax.legend(fontsize=12); ax.grid(True,alpha=0.3); ax.set_ylim(bottom=0)
        plt.tight_layout()
        fn_=os.path.join(OUT,f'convergence_{title.replace("=","")}.png')
        plt.savefig(fn_,dpi=150); pr(f"  Saved {fn_}")

    # ── Bar plots ──
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    xp=np.arange(3); w=0.18
    ax=axes[0]
    for i,s0 in enumerate(s0_list):
        ref=refs2[s0]
        kw1={'label':'VM UB'} if i==0 else {}; kw2={'label':'PO UB'} if i==0 else {}
        kw3={'label':'LB'} if i==0 else {}
        ax.bar(xp[i]-w,R[('VM',2,s0)]['LB'].mean(),w,color='#66BB6A',**kw3)
        ax.bar(xp[i],R[('VM',2,s0)]['Vt'].mean(),w,color='#1565C0',**kw1)
        ax.bar(xp[i]+w,R[('PO',2,s0)]['Vt'].mean(),w,color='#E65100',**kw2)
        ax.hlines([ref[0],ref[1]],xp[i]-1.5*w,xp[i]+1.5*w,colors='r',linestyles='--',lw=1.5)
    ax.set_xticks(xp); ax.set_xticklabels([f's0={s}' for s in s0_list])
    ax.set_ylabel('Price'); ax.set_title('Bounds Comparison (d=2)'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.3)

    ax=axes[1]
    vg=[R[('VM',2,s)]['Vt'].mean()-R[('VM',2,s)]['LB'].mean() for s in s0_list]
    pg=[R[('PO',2,s)]['Vt'].mean()-R[('PO',2,s)]['LB'].mean() for s in s0_list]
    ax.bar(xp-.15,vg,.3,color='#1565C0',label='VM'); ax.bar(xp+.15,pg,.3,color='#E65100',label='PO')
    ax.set_xticks(xp); ax.set_xticklabels([f's0={s}' for s in s0_list])
    ax.set_ylabel('Gap'); ax.set_title('Duality Gap (d=2)'); ax.legend(); ax.grid(axis='y',alpha=.3)

    ax=axes[2]
    ax.bar(xp-.15,[R[('VM',2,s)]['Vt'].std() for s in s0_list],.3,color='#1565C0',label='VM')
    ax.bar(xp+.15,[R[('PO',2,s)]['Vt'].std() for s in s0_list],.3,color='#E65100',label='PO')
    ax.set_xticks(xp); ax.set_xticklabels([f's0={s}' for s in s0_list])
    ax.set_ylabel('Std'); ax.set_title('Std of Upper Bounds (d=2)'); ax.legend(); ax.grid(axis='y',alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'comparison_plots.png'),dpi=150)

    # ── Save text results ──
    with open(os.path.join(OUT,'results_comparison.txt'),'w') as f:
        f.write(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"N_rep={N_rep}\n")
        for dd,refs in [(2,refs2),(5,refs5)]:
            f.write(f"\nd={dd}\n{'='*120}\n")
            f.write(f"{'s0':>4} | {'V0_tilde':>8}({'std':>5}) | {'V0_cap':>8}({'std':>5}) | {'95% CI':>22} | Method\n")
            f.write(f"{'-'*80}\n")
            for s0 in s0_list:
                if ('VM',dd,s0) not in R: continue
                vm=R[('VM',dd,s0)]; po=R[('PO',dd,s0)]; ref=refs[s0]
                vm_lb_m,vm_lb_s=vm['LB'].mean(),vm['LB'].std()
                vm_ub_m,vm_ub_s=vm['Vt'].mean(),vm['Vt'].std()
                vm_ci_lo=vm_lb_m-1.96*vm_lb_s/np.sqrt(N_rep)
                vm_ci_hi=vm_ub_m+1.96*vm_ub_s/np.sqrt(N_rep)
                po_lb_m,po_lb_s=po['LB'].mean(),po['LB'].std()
                po_ub_m,po_ub_s=po['Vt'].mean(),po['Vt'].std()
                po_ci_lo=po_lb_m-1.96*po_lb_s/np.sqrt(N_rep)
                po_ci_hi=po_ub_m+1.96*po_ub_s/np.sqrt(N_rep)
                f.write(f"{s0:>4} | {vm_lb_m:8.2f}({vm_lb_s:5.2f}) | {vm_ub_m:8.2f}({vm_ub_s:5.2f}) | [{vm_ci_lo:8.2f},{vm_ci_hi:8.2f}] | VM  Ref:[{ref[0]},{ref[1]}]\n")
                f.write(f"{s0:>4} | {po_lb_m:8.2f}({po_lb_s:5.2f}) | {po_ub_m:8.2f}({po_ub_s:5.2f}) | [{po_ci_lo:8.2f},{po_ci_hi:8.2f}] | PO\n")

    pr(f"\nAll files saved. TOTAL RUNTIME: {total_time:.1f}s ({total_time/60:.1f} min)")
