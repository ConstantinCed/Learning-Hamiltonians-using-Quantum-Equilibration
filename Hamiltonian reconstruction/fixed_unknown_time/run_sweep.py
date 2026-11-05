"""Fixed-time spectral learning and exact dynamical gaps.

Run python3 run_sweep.py, then python3 make_plot.py. Defaults: n=6,8,10,
100000 total shots per (Hamiltonian,time), eight seeds, 13 logarithmic times
from 0.01 to 10000. Each reconstruction uses ONE fixed time hidden from the
learner. Input/basis schedules are paired across times; Born outcomes use
independent random streams. Every shot prepares a fresh random product input.
Raw records and provenance are stored under output/fixed_time_recovery/.
"""
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy
from scipy.linalg import eigh, null_space

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'unknown_and_varying_time'))
import simulate as base
ROTATIONS=np.array(base.rc.BASIS_ROT_LIST)


def sample_outcomes(state,axes,uniforms):
    """Exact joint Born sample via sequential conditional probabilities.

This is an inverse-CDF sampler for one simultaneous product-basis measurement,
not repeated evolution or a measurement-backaction trajectory. Conditional
states halve in dimension at each qubit, reducing classical simulation cost.
"""
    state=state.copy();u=uniforms.copy()
    outcomes=np.zeros(len(u),dtype=np.uint16)
    norm_error=float(np.max(abs(np.sum(abs(state)**2,axis=0)-1)))
    assert norm_error<1e-10
    for q in range(axes.shape[1]):
        a,b=np.split(state,2,axis=0);R=ROTATIONS[axes[:,q]]
        zero=R[:,0,0]*a+R[:,0,1]*b
        one=R[:,1,0]*a+R[:,1,1]*b
        p0=np.sum(abs(zero)**2,axis=0)
        p1=np.sum(abs(one)**2,axis=0)
        probability=p0/(p0+p1)
        bit=u>=probability
        outcomes=(outcomes<<1)|bit.astype(np.uint16)
        denominator=np.where(bit,1-probability,probability)
        assert np.all(denominator>0)
        u=(u-np.where(bit,probability,0))/denominator
        # Roundoff at a branch boundary must not create an invalid uniform.
        u=np.clip(u,0,np.nextafter(1.,0.))
        state=np.where(bit[None,:],one,zero)
    return outcomes,norm_error


def commutator_gram(U,descriptors):
    """G_ab=Re Tr([U,P_a]^dagger [U,P_b])/d, computed exactly."""
    d=len(U);idx=np.arange(d)
    real=np.empty((len(descriptors),d*d));imag=np.empty_like(real)
    for k,(flip,phase) in enumerate(descriptors):
        C=U[:,idx^flip]*phase[None,:]-phase[idx^flip,None]*U[idx^flip,:]
        real[k]=C.real.ravel();imag[k]=C.imag.ravel()
    G=(real@real.T+imag@imag.T)/d
    return (G+G.T)/2


def learn(matrix_sum,shots):
    M=matrix_sum/shots
    values,vectors=eigh((M+M.T)/2)
    return vectors[:,-1],float(values[-1]-values[-2])


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def simulate_point(U,codes,axes,uniforms,batch_size):
    terms=9*(codes.shape[1]-1)
    accum=np.zeros((terms,terms));outcomes=np.empty(len(codes),dtype=np.uint16)
    norm_error=0.
    for lo in range(0,len(codes),batch_size):
        hi=min(lo+batch_size,len(codes))
        state=U@base.product_states(codes[lo:hi])
        observations,error=sample_outcomes(state,axes[lo:hi],uniforms[lo:hi])
        outcomes[lo:hi]=observations;norm_error=max(norm_error,error)
        X,Y=base.statistics(codes[lo:hi],axes[lo:hi],observations)
        accum+=Y.T@X
    estimate,empirical_gap=learn(accum,len(codes))
    return estimate,empirical_gap,accum,outcomes,norm_error


def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    times=np.logspace(-2,4,13) if args.times is None else np.array(args.times,float)
    config=dict(sizes=args.sizes,shots=args.shots,instances=args.instances,
        first_seed=args.first_seed,times=times.tolist(),batch_size=args.batch_size)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(base.__file__),Path(base.rc.__file__)]}
    metadata=dict(config=config,source_sha256=hashes,
        model='All adjacent two-body Pauli terms on an open chain; iid Gaussian coefficients, no one-body terms',
        normalization='H=sqrt(n)*sum h_a P_a; ||h||_2=1',
        budget='100000 fresh input-evolution-measurement shots per Hamiltonian and time by default; no time labels supplied to learner',
        seeds='Hamiltonian: seed; inputs: SeedSequence([n,seed,10000]); axes: [n,seed,70000]; outcomes: [n,seed,80000,time_index]',
        pairing='Same Hamiltonian, input states and measurement axes at all plotted times; independent Born random numbers at each time',
        diagnostic='Exact pi_U=lambda_min(G restricted to h-perp), G_ab=Re Tr([U,P_a]^dagger[U,P_b])/2^n; gamma=pi_U/2',
        error='min_{sign=+-1} ||h_hat-sign*h||_2, with unit-norm coefficients',
        numpy=np.__version__,scipy=scipy.__version__,python=sys.version)
    metapath=dest/'metadata.json'
    if metapath.exists():
        old=json.loads(metapath.read_text())
        assert old['config']==config and old['source_sha256']==hashes, 'Changed configuration or code: choose a new output directory.'
    else:metapath.write_text(json.dumps(metadata,indent=2)+'\n')
    all_rows=[]
    for n in args.sizes:
        for seed in range(args.first_seed,args.first_seed+args.instances):
            start=time.time()
            labels,H,h,descriptors,E,V=base.model(n,seed)
            Q=null_space(h[None,:])
            codes=np.random.default_rng(np.random.SeedSequence([n,seed,10000])).integers(0,6,(args.shots,n))
            axes=np.random.default_rng(np.random.SeedSequence([n,seed,70000])).integers(0,3,(args.shots,n))
            schedule=dest/f'schedule_n{n}_seed{seed}.npz'
            if not schedule.exists():
                np.savez_compressed(schedule,codes=codes.astype(np.uint8),axes=axes.astype(np.uint8),h=h,labels=np.array(labels))
            for k,t in enumerate(times):
                path=dest/f'n{n}_seed{seed}_time{k:02d}.npz'
                rowpath=dest/f'n{n}_seed{seed}_time{k:02d}.csv'
                if path.exists() and rowpath.exists():
                    with rowpath.open() as f:all_rows.extend(csv.DictReader(f))
                    continue
                point_start=time.time()
                U=(V*np.exp(-1j*E*t))@V.conj().T
                G=commutator_gram(U,descriptors)
                projected=Q.T@G@Q
                pi_U=float(eigh(projected,eigvals_only=True,subset_by_index=[0,0])[0])
                conservation=float(np.linalg.norm(G@h))
                assert pi_U>=-1e-10 and pi_U<=4+1e-10
                assert conservation<1e-10
                uniforms=np.random.default_rng(np.random.SeedSequence([n,seed,80000,k])).random(args.shots)
                estimate,empirical_gap,accum,outcomes,norm_error=simulate_point(U,codes,axes,uniforms,args.batch_size)
                if estimate@h<0:estimate=-estimate
                row=dict(n=n,seed=seed,time_index=k,time=t,total_shots=args.shots,
                    **base.errors(estimate,h),pi_U=pi_U,spectral_gap=pi_U/2,
                    max_non_h_autocorrelation=1-pi_U/2,empirical_gap=empirical_gap,
                    conservation_residual=conservation,probability_norm_error=norm_error)
                np.savez_compressed(path,time=t,estimate=estimate,matrix_sum=accum,
                    outcomes=outcomes,commutator_gram=G)
                write_csv(rowpath,[row]);all_rows.append(row)
                print(f'n={n}, seed={seed}, t={t:.5g}: error={row["direction_error"]:.4f}, F={100*row["fidelity"]:.2f}%, pi={pi_U:.5g} ({time.time()-point_start:.1f}s)',flush=True)
            print(f'Completed n={n}, seed={seed} in {time.time()-start:.1f}s',flush=True)
    write_csv(dest/'metrics.csv',all_rows)
    print(f'COMPLETE: {len(all_rows)} reconstructions, {args.shots} shots each.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',nargs='+',type=int,default=[6,8,10])
    p.add_argument('--shots',type=int,default=100000)
    p.add_argument('--instances',type=int,default=8)
    p.add_argument('--first-seed',type=int,default=123)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--times',nargs='+',type=float)
    p.add_argument('--output',default=str(HERE.parents[1]/'output/fixed_time_recovery'))
    run(p.parse_args())
