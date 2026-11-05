"""Audit million-shot checkpoints and compare with the original 100k prefixes."""
import csv
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, null_space
import run_sweep as sim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    data = ROOT / 'output/fixed_time_recovery_1m'
    original = ROOT / 'output/fixed_time_recovery'
    have_original = (original / 'metadata.json').exists()
    with (data / 'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 312
    schedules = {}
    for n in [6, 8, 10]:
        for seed in range(123, 131):
            filename = f'schedule_n{n}_seed{seed}.npz'
            with np.load(data / f'seed_{seed}' / filename) as current:
                assert current['codes'].shape == (1000000, n)
                assert current['axes'].shape == (1000000, n)
                if have_original:
                    with np.load(original / filename) as old:
                        np.testing.assert_array_equal(current['codes'][:100000], old['codes'])
                        np.testing.assert_array_equal(current['axes'][:100000], old['axes'])
                        np.testing.assert_array_equal(current['h'], old['h'])
                schedules[n, seed] = current['h']
    for row in rows:
        n, seed, k = (int(row[key]) for key in ['n', 'seed', 'time_index'])
        filename = f'n{n}_seed{seed}_time{k:02d}.npz'
        h = schedules[n, seed]
        with np.load(data / f'seed_{seed}' / filename) as record:
            assert record['outcomes'].shape == (1000000,)
            assert np.max(record['outcomes']) < 2**n
            assert float(record['time']) == float(row['time'])
            # When available, the previous experiment must be an exact prefix.
            if have_original:
                with np.load(original / filename) as old:
                    np.testing.assert_array_equal(record['outcomes'][:100000], old['outcomes'])
                    np.testing.assert_allclose(record['commutator_gram'], old['commutator_gram'], atol=1e-12)
            vector, _ = sim.learn(record['matrix_sum'], 1000000)
            vector *= 1 if vector @ h >= 0 else -1
            np.testing.assert_allclose(vector, record['estimate'], atol=1e-12)
            assert abs(np.linalg.norm(vector - h) - float(row['direction_error'])) < 1e-12
            assert abs((vector @ h)**2 - float(row['fidelity'])) < 1e-12
            Q = null_space(h[None, :])
            gap = eigh(Q.T @ record['commutator_gram'] @ Q, eigvals_only=True)[0]
            assert abs(gap - float(row['pi_U'])) < 1e-11
            assert np.linalg.norm(record['commutator_gram'] @ h) < 1e-10
    # Independently reconstruct the sufficient statistic from raw records.
    for n in [6, 8, 10]:
        directory = data / 'seed_123'
        with np.load(directory / f'schedule_n{n}_seed123.npz') as schedule, np.load(directory / f'n{n}_seed123_time12.npz') as record:
            codes, axes, outcomes = schedule['codes'], schedule['axes'], record['outcomes']
            accum = np.zeros((9*(n-1), 9*(n-1)), dtype=np.int64)
            for lo in range(0, len(codes), 4096):
                c = codes[lo:lo+4096].astype(np.int64)
                a = axes[lo:lo+4096].astype(np.int64)
                o = outcomes[lo:lo+4096].astype(np.int64)
                basis, signs = c//2, 1-2*(c%2)
                bits = 1-2*((o[:, None] >> np.arange(n-1, -1, -1)) & 1)
                xidx = 9*np.arange(n-1)+3*basis[:, :-1]+basis[:, 1:]
                yidx = 9*np.arange(n-1)+3*a[:, :-1]+a[:, 1:]
                xsign = signs[:, :-1]*signs[:, 1:]
                ysign = bits[:, :-1]*bits[:, 1:]
                np.add.at(accum, (yidx[:, :, None], xidx[:, None, :]),
                    81*ysign[:, :, None]*xsign[:, None, :])
            np.testing.assert_array_equal(accum, record['matrix_sum'])
    print('PASS: all 312 estimates, gaps and budgets; three independently accumulated million-shot records.')
    print('Exact 100k prefixes: verified for all 312 records.' if have_original else
          'Original 100k raw records unavailable; prefix comparison skipped.')


if __name__ == '__main__':
    main()
