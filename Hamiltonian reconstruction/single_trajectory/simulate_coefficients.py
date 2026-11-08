"""Reconstruct a Hamiltonian from one simulated measurement trajectory (Fig. S5).

python3 simulate_coefficients.py            # replay and check the published run
python3 simulate_coefficients.py --ensemble # regenerate the CSV from all 24 runs

Requires NumPy and SciPy. The portable Python replay is slower than replotting
the CSV. Each run prepares a single product state; the measured product state
is the next input, including between blocks. The intervals are independent
Uniform[400,800] draws used only by the Born simulator, not the learner.
"""
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

CSV = Path(__file__).with_name('trajectory_n10_coefficients.csv')
BUDGETS = (2000,5000,10000,20000,50000,100000,200000,500000)
SQ=np.sqrt(.5)
LOCAL=np.array([[SQ,SQ],[SQ,-SQ],[SQ,1j*SQ],[SQ,-1j*SQ],[1,0],[0,1]],complex)
ROT=np.array([LOCAL[2*q:2*q+2].conj() for q in range(3)])

def product(codes):
    a=np.ones(1,complex)
    for c in codes:a=np.kron(a,LOCAL[c])
    return a

def descriptors(n):
    # P|j> = phase[j] |j xor flip>
    j=np.arange(1<<n);ret=[]
    for bond in range(n-1):
        for a,b in itertools.product(range(3),repeat=2):
            flip=0;phase=np.ones(len(j),complex);label=['I']*n
            for site,axis in [(bond,a),(bond+1,b)]:
                mask=1<<(n-1-site);sgn=1-2*((j&mask)!=0).astype(int)
                if axis<2:flip^=mask
                if axis==1:phase*=1j*sgn
                if axis==2:phase*=sgn
                label[site]='XYZ'[axis]
            ret.append((''.join(label),flip,phase))
    return ret

def model(n,seed):
    desc=descriptors(n);g=np.random.default_rng(seed).normal(size=len(desc))
    h=np.sqrt(n)*g/np.linalg.norm(g);idx=np.arange(1<<n)
    H=np.zeros((len(idx),len(idx)),complex)
    for c,(_,flip,p) in zip(h,desc):H[idx^flip,idx]+=c*p
    np.testing.assert_allclose(H,H.conj().T,atol=1e-14)
    np.testing.assert_allclose(np.vdot(H,H).real/len(idx),n,atol=1e-12)
    E,V=eigh(H)
    return h,H,E,V

def features(records):
    a=records.astype(np.int64);q=a//2;s=1-2*(a%2);n=a.shape[1]
    f=np.zeros((len(a),9*(n-1)))
    f[np.arange(len(a))[:,None],9*np.arange(n-1)+3*q[:,:-1]+q[:,1:]]=9*s[:,:-1]*s[:,1:]
    return f

def learn(matrix_sum,m):
    # No time, true Hamiltonian, protocol or exact gap is available here.
    S=(matrix_sum+matrix_sum.T)/(2*m)
    v,W=eigh(S)
    return W[:,-1],float(v[-1]-v[-2])

def replay(seed,run,h,E,V,rounds=500000):
    n=10
    if not 0 < rounds <= BUDGETS[-1]:
        raise ValueError('Expected between 1 and 500000 rounds')
    def rng(field,lo=0):
        # Same SeedSequence fields as the study: timing=fluctuating (1),
        # method=trajectory (1). Each output axis and Born draw is independent.
        return np.random.default_rng(np.random.SeedSequence(
            [20261107,n,seed,1,run,1,field,lo]))
    state=rng(0).integers(0,6,n,dtype=np.uint8)
    shifts=np.arange(n-1,-1,-1)
    Vh=V.conj().T.copy()
    accum=np.zeros((len(h),len(h)),dtype=np.int64)
    lo=0
    for budget in BUDGETS:
        hi=min(budget,rounds)
        axes=rng(2,lo).integers(0,3,(hi-lo,n),dtype=np.uint8)
        uniforms=rng(3,lo).random(hi-lo)
        intervals=rng(4,lo).uniform(400,800,hi-lo)
        for start in range(0,hi-lo,4096):
            stop=min(start+4096,hi-lo)
            records=np.empty((stop-start+1,n),np.uint8)
            records[0]=state
            for j in range(start,stop):
                energy=Vh@product(state)
                psi=V@(np.exp(-1j*E*intervals[j])*energy)
                for q,axis in enumerate(axes[j]):
                    v=psi.reshape(1<<q,2,-1)
                    psi=np.einsum('ij,ajk->aik',ROT[axis],v).reshape(-1)
                prob=abs(psi)**2
                norm=prob.sum()
                if abs(norm-1)>1e-10:
                    raise ValueError('Born probability normalization failed')
                cdf=np.cumsum(prob/norm)
                cdf[-1]=1.
                word=np.searchsorted(cdf,uniforms[j],side='right')
                state=2*axes[j]+((word>>shifts)&1).astype(np.uint8)
                records[j-start+1]=state
            f=features(records)
            accum+=(f[1:].T@f[:-1]).astype(np.int64)
        if hi==rounds:break
        lo=hi
    v,_=learn(accum,rounds)
    F=float(np.clip((v@(h/np.linalg.norm(h)))**2,0.,1.))
    est=np.sqrt(n)*v*(1 if h@v>=0 else -1)
    return est,F,accum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ensemble',action='store_true')
    args=p.parse_args()
    if not args.ensemble:
        with CSV.open() as f:rows=list(csv.DictReader(f))
        first=rows[0];seed=int(first['hamiltonian_seed']);run=int(first['run'])
        h,_,E,V=model(10,seed)
        est,F,_=replay(seed,run,h,E,V,int(first['rounds']))
        np.testing.assert_allclose(h,[float(r['h_true']) for r in rows],atol=1e-12)
        np.testing.assert_allclose(est,[float(r['h_reconstructed']) for r in rows],rtol=0,atol=1e-10)
        np.testing.assert_allclose(F,float(first['fidelity']),rtol=0,atol=1e-12)
        print(f'Verified 81 coefficients; fidelity = {F:.8f}')
        return
    results=[]
    for seed in range(123,135):
        h,_,E,V=model(10,seed)
        for run in range(2):
            est,F,_=replay(seed,run,h,E,V)
            results.append((seed,run,h,est,F))
    median=float(np.median([r[-1] for r in results]))
    seed,run,h,est,F=min(results,key=lambda r:(round(abs(r[-1]-median),14),r[0],r[1]))
    labels=[s[0] for s in descriptors(10)]
    rows=[dict(index=i,bond=i//9+1,label=label,h_true=float(h[i]),h_reconstructed=float(est[i]),
               residual=float(est[i]-h[i]),hamiltonian_seed=seed,run=run,timing='fluctuating',
               rounds=500000,fidelity=F,selection_median_fidelity=median)
          for i,label in enumerate(labels)]
    with CSV.open('w',newline='') as f:
        writer=csv.DictWriter(f,rows[0].keys(),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)

if __name__=='__main__':
    main()
