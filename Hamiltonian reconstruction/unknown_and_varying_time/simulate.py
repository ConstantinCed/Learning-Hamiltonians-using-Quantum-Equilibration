"""Single-shot spectral recovery with unknown, increasing evolution durations.

Run: python3 simulate.py --total-shots 100000
Default: 10 qubits, 12 seeds (123-134), one independent six-state product input
and one independent local-Pauli measurement per shot. Raw records are saved to
raw_results/. Times are hidden from the spectral learner. This is a numerical
experiment for correlated increasing times, not a proof of recovery guarantees.
The included figure uses 100k shots and chronological prefixes of each record.
Rebuild the included figure with python3 make_plot.py (requires LaTeX and Poppler).
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
from scipy.linalg import eigh

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import recon_common as rc
PREFIXES=[1000,2000,5000,10000,20000,50000,100000,150000,200000]


def model(n, seed):
    labels = [rc.two_site_label(n, i, i+1, p, q)
              for i in range(n-1) for p in "XYZ" for q in "XYZ"]
    raw = np.random.default_rng(seed).normal(size=len(labels))
    H, h, descriptors = rc.build_hamiltonian(labels, raw)
    H *= np.sqrt(n)/np.linalg.norm(raw)
    E, V = eigh(H)
    return labels, H, h, descriptors, E, V

def errors(v, h):
    overlap = float(np.clip(abs(v@h),0,1))
    return dict(direction_error=float(np.linalg.norm(v-h)),
                coefficient_rmse=float(np.sqrt(np.mean((v-h)**2))),
                coefficient_mae=float(np.mean(abs(v-h))),
                coefficient_max_error=float(np.max(abs(v-h))),
                fidelity=overlap**2, infidelity=1-overlap**2)

def save_csv(path, rows):
    with path.open("w",newline="") as f:
        w = csv.DictWriter(f,list(rows[0])); w.writeheader(); w.writerows(rows)


def records(n,total,seed,t0=1000.,low=.025,high=.075):
    # Separate RNG streams and fixed-width integer draws preserve prefixes.
    codes=np.random.default_rng(seed+10000).integers(0,6,(total,n))
    axes=np.random.default_rng(seed+70000).integers(0,3,(total,n))
    increments=np.random.default_rng(seed+50000).uniform(low,high,total)
    uniforms=np.random.default_rng(seed+80000).random(total)
    return codes,axes,increments,t0+np.cumsum(increments),uniforms


def product_states(codes):
    local=np.array([p[2] for p in rc.PROBE_OPTIONS])
    psi=np.ones((1,len(codes)),complex)
    for q in range(codes.shape[1]):
        psi=(psi[:,None,:]*local[codes[:,q]].T[None,:,:]).reshape(-1,len(codes))
    return psi


def rotate_local(psi,axes):
    """Apply a different tensor-product measurement basis to every column."""
    n=axes.shape[1];out=psi.copy()
    for q in range(n):
        block=out.reshape(2**q,2,2**(n-q-1),len(axes))
        a=block[:,0,:,:].copy();b=block[:,1,:,:].copy()
        x=axes[:,q]==0;y=axes[:,q]==1
        block[:,0,:,x]=((a+b)/np.sqrt(2))[:,:,x].transpose(2,0,1)
        block[:,1,:,x]=((a-b)/np.sqrt(2))[:,:,x].transpose(2,0,1)
        block[:,0,:,y]=((a-1j*b)/np.sqrt(2))[:,:,y].transpose(2,0,1)
        block[:,1,:,y]=((a+1j*b)/np.sqrt(2))[:,:,y].transpose(2,0,1)
    return out


def sample_batch(E,V,codes,axes,times,uniforms):
    energy=V.conj().T@product_states(codes)
    state=V@(energy*np.exp(-1j*E[:,None]*times[None,:]))
    amplitude=rotate_local(state,axes)
    prob=(abs(amplitude)**2).T
    norm_error=float(np.max(abs(prob.sum(axis=1)-1)))
    assert norm_error<1e-10
    prob/=prob.sum(axis=1,keepdims=True)
    cdf=np.cumsum(prob,axis=1);cdf[:,-1]=1.
    return np.sum(cdf<uniforms[:,None],axis=1).astype(np.uint16),norm_error


def statistics(codes,axes,outcomes):
    n=codes.shape[1];batch=len(codes);rows=np.arange(batch)[:,None]
    bonds=np.arange(n-1)[None,:]
    bases=codes//2;signs=1-2*(codes%2)
    inp=np.zeros((batch,9*(n-1)))
    inp[rows,9*bonds+3*bases[:,:-1]+bases[:,1:]]=9*signs[:,:-1]*signs[:,1:]
    bits=1-2*((outcomes[:,None].astype(np.int64)>>np.arange(n-1,-1,-1))&1)
    out=np.zeros_like(inp)
    out[rows,9*bonds+3*axes[:,:-1]+axes[:,1:]]=9*bits[:,:-1]*bits[:,1:]
    return inp,out


def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    prefixes=sorted(set([m for m in PREFIXES if m<=args.total_shots]+[args.total_shots]))
    config=dict(n=args.n,total_shots=args.total_shots,shots_per_probe=1,
        probe_protocol='fresh_independent_each_shot',instances=args.instances,
        first_seed=args.first_seed,t0=args.t0,increment_low=.025,increment_high=.075,
        prefix_shots=prefixes)
    scientific_config={k:v for k,v in config.items() if k not in ['total_shots','prefix_shots']}
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    common_hash=hashlib.sha256(Path(rc.__file__).read_bytes()).hexdigest()
    metadata_path=dest/'metadata.json'
    if metadata_path.exists():
        old=json.loads(metadata_path.read_text())
        assert old['scientific_config']==scientific_config
        assert old['source_sha256']==source_hash and old['common_source_sha256']==common_hash
        assert old['config']['total_shots']<=args.total_shots
    meta=dict(config=config,scientific_config=scientific_config,
        seeds=list(range(args.first_seed,args.first_seed+args.instances)),
        model='All 9 adjacent two-body Pauli terms per bond, open chain, iid Gaussian coefficients; no one-body terms',
        normalization='H=sqrt(n)*sum h_a P_a, norm(h)=1',
        probe='Every shot independently samples each qubit from the six Pauli eigenstates; coincidental repeated inputs are allowed',
        measurement='Every shot independently samples a uniform Pauli axis on every qubit; one joint bit-string measurement',
        estimator='M_hat = mean(Y X^T); X=9*known two-body input expectation; Y=9*outcome product if both axes match, else 0; top eigenvector of symmetric part',
        timing='One global increasing duration trajectory; fresh preparation each shot; times and their distribution are hidden from the learner',
        seed_rules=dict(probes='seed+10000',times='seed+50000',axes='seed+70000',outcomes='seed+80000'),
        source_sha256=source_hash,common_source_sha256=common_hash,
        numpy=np.__version__,scipy=scipy.__version__,python=sys.version)
    metadata_path.write_text(json.dumps(meta,indent=2)+'\n')
    for seed in meta['seeds']:
        start_time=time.time();path=dest/f'instance_{seed}.npz'
        codes,axes,increments,times,uniforms=records(args.n,args.total_shots,seed,args.t0)
        saved={};metrics=[];begin=0
        if path.exists():
            with np.load(path) as old:saved={k:old[k] for k in old.files}
            begin=len(saved['outcomes'])
            assert begin<=args.total_shots
            for key,value in [('codes',codes),('axes',axes),('times',times),('increments',increments)]:
                assert np.array_equal(saved[key],value[:begin])
            if begin==args.total_shots:
                print(f'Reusing seed {seed} at {begin} shots',flush=True);continue
            with (dest/f'metrics_{seed}.csv').open() as f:metrics=list(csv.DictReader(f))
        labels,H,h,descriptors,E,V=model(args.n,seed)
        accum=saved.get('matrix_sum',np.zeros((len(h),len(h))))
        outcomes=np.empty(args.total_shots,dtype=np.uint16)
        if begin:outcomes[:begin]=saved['outcomes']
        norm_error=float(saved.get('max_norm_error',0.))
        for end in prefixes:
            if end<=begin:continue
            for lo in range(begin,end,args.batch_size):
                hi=min(lo+args.batch_size,end)
                obs,err=sample_batch(E,V,codes[lo:hi],axes[lo:hi],times[lo:hi],uniforms[lo:hi])
                outcomes[lo:hi]=obs;norm_error=max(norm_error,err)
                X,Y=statistics(codes[lo:hi],axes[lo:hi],obs)
                accum+=Y.T@X
            M=accum/end;S=(M+M.T)/2
            eigenvalues,eigenvectors=eigh(S);v=eigenvectors[:,-1]
            if v@h<0:v=-v
            saved[f'spectral_{end}']=v
            metrics.append(dict(seed=seed,total_shots=end,shots_per_probe=1,method='spectral',
                **errors(v,h),t_first=times[0],t_last=times[end-1],
                probability_norm_error=norm_error,empirical_top_eigenvalue=eigenvalues[-1],
                empirical_gap=eigenvalues[-1]-eigenvalues[-2]))
            print(f"Seed {seed}, M={end}: F={errors(v,h)['fidelity']:.6f}",flush=True)
            begin=end
        saved.update(h=h,labels=np.array(labels),codes=codes.astype(np.uint8),axes=axes.astype(np.uint8),
            increments=increments,times=times,outcomes=outcomes,matrix_sum=accum,max_norm_error=norm_error)
        np.savez_compressed(path,**saved)
        save_csv(dest/f'metrics_{seed}.csv',metrics)
        v=saved[f'spectral_{args.total_shots}']
        save_csv(dest/f'coefficients_{seed}.csv',[dict(seed=seed,index=i,label=label,true_h=h[i],spectral=v[i]) for i,label in enumerate(labels)])
        print(f'Completed seed {seed} in {time.time()-start_time:.1f}s',flush=True)
    for name in ['metrics','coefficients']:
        rows=[]
        for seed in meta['seeds']:
            with (dest/f'{name}_{seed}.csv').open() as f:rows.extend(csv.DictReader(f))
        save_csv(dest/f'{name}.csv',rows)
    with (dest/'metrics.csv').open() as f:all_metrics=list(csv.DictReader(f))
    final=[float(r['fidelity']) for r in all_metrics if int(r['total_shots'])==args.total_shots]
    print(f'FINAL: {args.total_shots} total shots, median fidelity {100*np.median(final):.6f}%, IQR {100*np.percentile(final,[25,75])}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=10)
    p.add_argument('--total-shots',type=int,default=100000)
    p.add_argument('--instances',type=int,default=12)
    p.add_argument('--first-seed',type=int,default=123)
    p.add_argument('--t0',type=float,default=1000.)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--output',default=str(HERE/'raw_results'))
    run(p.parse_args())
