"""Matched fresh-preparation arm for Figure 3b. Run with python3 matched_baseline.py.

Uses the same Gaussian Hamiltonians, two replicates, Uniform[400,800] interval
law and round budgets as the single-trajectory study. The fresh arm has its
own preparation, timing, measurement-axis and Born RNG streams. Each round
starts from a new independent product state, with one evolution and one output
measurement. Only the empirical cross-correlation enters the learner.
"""
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,sys,concurrent.futures,time
from pathlib import Path
import numpy as np
from simulate_coefficients import model,features,learn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from unknown_and_varying_time.simulate import sample_batch
BUDGETS=(2000,5000,10000,20000,50000,100000,200000,500000)
DEFAULT=HERE.parents[1]/'output/fig3_matched_baseline'
def run(args):
    seed,root,budgets,batch=args; n=10
    h,_,E,V=model(n,seed); rows=[]
    for rep in range(2):
        accum=np.zeros((81,81),np.int64);lo=0
        for hi in budgets:
            def rng(field):
                return np.random.default_rng(np.random.SeedSequence([20261107,n,seed,1,rep,0,field,lo]))
            k=hi-lo
            codes=rng(1).integers(0,6,(k,n),dtype=np.uint8)
            axes=rng(2).integers(0,3,(k,n),dtype=np.uint8)
            u=rng(3).random(k);intervals=rng(4).uniform(400,800,k)
            worst=0.
            for start in range(0,k,batch):
                z=slice(start,start+batch)
                words,err=sample_batch(E,V,codes[z],axes[z],intervals[z],u[z])
                outputs=2*axes[z]+((words[:,None].astype(int)>>np.arange(n-1,-1,-1))&1)
                accum+=(features(outputs).T@features(codes[z])).astype(np.int64)
                worst=max(worst,err)
            v,gap=learn(accum,hi)
            rows.append(dict(seed=seed,run=rep,rounds=hi,preparations=hi,
                fidelity=float((v@(h/np.linalg.norm(h)))**2),empirical_gap=gap,
                max_block_norm_error=worst))
            lo=hi
    output=root/f'seed_{seed}.csv'
    with output.open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    return seed
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--seeds',nargs='+',type=int,default=list(range(123,135)))
    p.add_argument('--shots',type=int,choices=BUDGETS,default=500000)
    p.add_argument('--batch',type=int,default=512);p.add_argument('--output',type=Path,default=DEFAULT)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    t=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(run,(s,a.output,[m for m in BUDGETS if m<=a.shots],a.batch)) for s in a.seeds]
        for job in concurrent.futures.as_completed(jobs):
            print(f'Seed {job.result()} complete through {a.shots} rounds per replicate ({time.perf_counter()-t:.1f}s)',flush=True)
