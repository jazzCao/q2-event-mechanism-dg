"""Analytic software checks, not additional bearing experiments."""
import unittest
import numpy as np
from q2.operator import edit_context, query_features, Q2_RATES
from q2.renderer import render
from q2.inverse import kernel
from q2.sampling import shared_edit_mask, draw_batch


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.s=np.zeros((2,2048));self.s[0,[450,900,1400]]=[1,.7,1.2];self.s[1,[650,1200]]=[.6,.8]
        self.p=np.array([[.10,.03,0.],[.18,.02,np.pi/2]])
        self.length=96;self.x=render(self.s,self.p,1.,self.length)+.1*np.sin(np.arange(2048)*.01)

    def test_exact_identity(self):
        np.testing.assert_array_equal(edit_context(self.x,self.s,self.p,2.4,1.,self.length),self.x)

    def test_independent_direct_kernel_sum(self):
        for q in (1.,*Q2_RATES):
            direct=np.zeros(2048);center=2047/2
            for i,p in enumerate(self.p):
                for pos in np.flatnonzero(self.s[i]):
                    direct+=self.s[i,pos]*kernel(np.arange(2048)-(center+(pos-center)/q),p,self.length)
            np.testing.assert_allclose(render(self.s,self.p,q,self.length),direct,rtol=1e-10,atol=1e-11)

    def test_remainder_is_preserved(self):
        u=render(self.s,self.p,1.,self.length)
        for q in Q2_RATES:
            edited=edit_context(self.x,self.s,self.p,1.7,q,self.length)
            uq=render(self.s,self.p,q,self.length)
            np.testing.assert_allclose(edited-1.7*q*uq,self.x-1.7*u,rtol=1e-12,atol=1e-12)

    def test_full_support_and_features(self):
        output=query_features(self.x,self.s,self.p,1.,[800],128,self.length)
        self.assertEqual(output['coupled'].shape,(1,2,2048))
        self.assertEqual(output['clean'].dtype,np.float32)
        with self.assertRaises(ValueError):query_features(self.x,self.s,self.p,1.,[0],128,self.length)

    def test_mask_shared_across_domains(self):
        u=np.arange(24)/24;mask=shared_edit_mask(u,2,3,4).reshape(2,3,4)
        np.testing.assert_array_equal(mask[0],mask[1])

    def test_sampler_state_replay(self):
        pools=[np.arange(i*4,(i+1)*4) for i in range(6)]
        s,a=np.random.RandomState(1),np.random.RandomState(2)
        state=(s.get_state(),a.get_state())
        one=draw_batch(pools,pools,2,2,3,s,a)
        s.set_state(state[0]);a.set_state(state[1]);two=draw_batch(pools,pools,2,2,3,s,a)
        for x,y in zip(one,two):np.testing.assert_equal(x,y)


if __name__=='__main__':unittest.main()
