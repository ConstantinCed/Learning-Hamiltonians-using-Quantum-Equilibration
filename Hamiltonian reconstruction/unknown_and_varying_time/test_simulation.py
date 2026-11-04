import itertools
import unittest
import numpy as np
from scipy.linalg import expm
import simulate as sim

class FreshProbeTests(unittest.TestCase):
    def test_shots_against_direct_exponentials(self):
        labels,H,h,desc,E,V=sim.model(3,12)
        codes,axes,delta,times,u=sim.records(3,100,71,t0=1.)
        actual,_=sim.sample_batch(E,V,codes,axes,times,u)
        expected=[]
        for c,a,t,r in zip(codes,axes,times,u):
            psi=sim.rc.kron_vec([sim.rc.PROBE_OPTIONS[k][2] for k in c])
            state=sim.rc.rotation_unitary(a)@expm(-1j*H*t)@psi
            prob=abs(state)**2;prob/=prob.sum()
            expected.append(np.searchsorted(np.cumsum(prob),r))
        np.testing.assert_array_equal(actual,expected)
        parts=[sim.sample_batch(E,V,codes[i:i+7],axes[i:i+7],times[i:i+7],u[i:i+7])[0] for i in range(0,100,7)]
        np.testing.assert_array_equal(np.concatenate(parts),actual)

    def test_exact_transfer_expectation(self):
        # Exhaust all input states, settings and outcomes; independently build
        # the transfer matrix from traces. This checks both importance weights.
        n=2;labels,H,h,desc,E,V=sim.model(n,12)
        U=expm(-1j*H*.73)
        P=[sim.rc.pauli_label_to_matrix(s) for s in labels]
        exact=np.array([[np.trace(U.conj().T@a@U@b).real/4 for b in P] for a in P])
        actual=np.zeros((9,9));second=np.zeros_like(actual)
        for code in itertools.product(range(6),repeat=n):
            c=np.tile(code,(4,1))
            psi=sim.rc.kron_vec([sim.rc.PROBE_OPTIONS[k][2] for k in code])
            for axes in itertools.product(range(3),repeat=n):
                a=np.tile(axes,(4,1))
                prob=abs(sim.rc.rotation_unitary(axes)@U@psi)**2
                X,Y=sim.statistics(c,a,np.arange(4))
                actual+=(Y.T*prob)@X/(36*9)
                second+=((Y**2).T*prob)@(X**2)/(36*9)
        np.testing.assert_allclose(actual,exact,atol=2e-13)
        np.testing.assert_allclose(second,81,atol=2e-11)

    def test_prefix_reproducibility(self):
        short=sim.records(10,1000,123);long=sim.records(10,2000,123)
        for a,b in zip(short,long):np.testing.assert_array_equal(a,b[:1000])
        self.assertTrue(np.all(np.diff(long[3])>0))
        np.testing.assert_allclose(long[3],1000+np.cumsum(long[2]))

if __name__=='__main__':unittest.main()
