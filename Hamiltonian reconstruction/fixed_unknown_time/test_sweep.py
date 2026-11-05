import unittest
import numpy as np
from scipy.linalg import expm,null_space,eigh
import run_sweep as sim


class SweepTests(unittest.TestCase):
    def test_joint_born_sampling(self):
        labels,H,h,desc,E,V=sim.base.model(3,123)
        codes,axes,_,_,u=sim.base.records(3,127,7)
        psi=sim.base.product_states(codes)
        for t in [0.,.01,1.,1000.]:
            U=expm(-1j*H*t);state=U@psi
            actual,_=sim.sample_outcomes(state,axes,u)
            expected=[]
            for j in range(len(codes)):
                rotated=sim.base.rc.rotation_unitary(axes[j])@state[:,j]
                prob=abs(rotated)**2;prob/=prob.sum()
                expected.append(np.searchsorted(np.cumsum(prob),u[j]))
            np.testing.assert_array_equal(actual,expected)

    def test_gap_matches_trace_transfer_matrix(self):
        labels,H,h,desc,E,V=sim.base.model(3,123)
        U=expm(-1j*H*.73)
        P=[sim.base.rc.pauli_label_to_matrix(s) for s in labels]
        M=np.array([[np.trace(U.conj().T@a@U@b).real/8 for b in P] for a in P])
        G=sim.commutator_gram(U,desc)
        np.testing.assert_allclose(G,2*np.eye(len(P))-M-M.T,atol=5e-14)
        np.testing.assert_allclose(G@h,0,atol=5e-14)
        Q=null_space(h[None,:])
        pi=eigh(Q.T@G@Q,eigvals_only=True)[0]
        correlation=eigh(Q.T@((M+M.T)/2)@Q,eigvals_only=True)[-1]
        self.assertAlmostEqual(pi,2*(1-correlation),places=12)
        estimate,_=sim.learn((M+M.T)/2,1)
        self.assertGreater(abs(estimate@h),1-1e-10)

    def test_identity_gap_and_batch_invariance(self):
        labels,H,h,desc,E,V=sim.base.model(3,8)
        np.testing.assert_array_equal(sim.commutator_gram(np.eye(8),desc),np.zeros((18,18)))
        codes,axes,_,_,u=sim.base.records(3,127,9)
        U=expm(-1j*H*.5)
        a=sim.simulate_point(U,codes,axes,u,17)
        b=sim.simulate_point(U,codes,axes,u,127)
        np.testing.assert_array_equal(a[3],b[3])
        np.testing.assert_array_equal(a[2],b[2])
        np.testing.assert_allclose(abs(a[0]@b[0]),1,atol=1e-14)


if __name__=='__main__':unittest.main()
