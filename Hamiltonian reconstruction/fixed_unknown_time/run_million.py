"""Repeat the fixed-time experiment with one million shots per reconstruction.

Run python3 run_million.py to simulate and rebuild the figure. To replot only,
run python3 make_plot.py --data ../../output/fixed_time_recovery_1m
(from this directory).
The eight original Hamiltonians and all 13 times are retained for every size.
Independent simulator processes write separate checkpoints; their complete
results are validated and combined without selecting successful instances.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main(workers):
    dest = HERE.parents[1] / 'output/fixed_time_recovery_1m'
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def instance(seed):
        directory = dest / f'seed_{seed}'
        with (dest / f'seed_{seed}.log').open('a') as log:
            subprocess.run([sys.executable, str(HERE / 'run_sweep.py'),
                '--shots', '1000000', '--instances', '1', '--first-seed', str(seed),
                '--output', str(directory)], env=env, stdout=log,
                stderr=subprocess.STDOUT, check=True)
        print(f'Completed seed {seed}: 39 reconstructions.', flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(instance, range(123, 131)))

    rows = []
    metadata = None
    for seed in range(123, 131):
        directory = dest / f'seed_{seed}'
        current = json.loads((directory / 'metadata.json').read_text())
        if metadata is None:
            metadata = current
        else:
            assert current['source_sha256'] == metadata['source_sha256']
        with (directory / 'metrics.csv').open() as f:
            batch = list(csv.DictReader(f))
        assert len(batch) == 39
        assert all(int(r['total_shots']) == 1000000 and int(r['seed']) == seed for r in batch)
        rows.extend(batch)
    assert len({(r['n'], r['seed'], r['time_index']) for r in rows}) == 312
    rows.sort(key=lambda r: (int(r['n']), int(r['seed']), int(r['time_index'])))
    metadata['config']['instances'] = 8
    metadata['budget'] = '1000000 fresh input-evolution-measurement shots per Hamiltonian and time; no time labels supplied to learner'
    metadata['checkpoint_layout'] = 'seed_{seed}/ contains schedules, raw outcomes, matrix sums and exact gaps'
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with (dest / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print('COMPLETE: 312 reconstructions, one million shots each.', flush=True)
    subprocess.run([sys.executable, str(HERE / 'verify_million.py')], check=True, env=env)
    subprocess.run([sys.executable, str(HERE / 'make_plot.py'), '--data', str(dest)], check=True, env=env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    main(parser.parse_args().workers)
