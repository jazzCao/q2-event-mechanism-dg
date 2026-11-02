"""Analytic synthetic cases for timing specificity and exact global moment controls."""
import unittest
import numpy as np
from event_component_core import impulse_only, global_affine_moments, factorial


class ComponentCoreTests(unittest.TestCase):
    def test_impulse_action_does_not_move_or_modify_remainder(self):
        u=np.zeros((2,12));u[0,2]=2.;u[1,8]=3.
        remainder=np.arange(24,dtype=float).reshape(2,12)/100
        f=remainder+u;v=impulse_only(f,u,1.25)
        np.testing.assert_allclose(v-remainder,1.25*u,rtol=0,atol=1e-15)
        np.testing.assert_array_equal(np.flatnonzero((v-f)[0]),[2])
        np.testing.assert_array_equal(np.flatnonzero((v-f)[1]),[8])

    def test_q1_is_exact_even_with_large_cancelling_response(self):
        f=np.array([[.1,.2,.3]]);u=np.array([[1e10,2e10,-3e10]])
        np.testing.assert_array_equal(impulse_only(f,u,1.),f)
        np.testing.assert_array_equal(global_affine_moments(f,f)[0],f)

    def test_affine_matches_both_moments_and_preserves_standardized_shape(self):
        f=np.array([[1.,2.,3.,4.],[-2.,0.,5.,7.]])
        target=np.array([[0.,0.,0.,8.],[1.,1.,2.,9.]])
        out,d=global_affine_moments(f,target)
        np.testing.assert_allclose(out.mean(1),target.mean(1),rtol=0,atol=1e-14)
        np.testing.assert_allclose(out.std(1),target.std(1),rtol=0,atol=1e-14)
        np.testing.assert_allclose(np.mean(out**2,1),np.mean(target**2,1),rtol=0,atol=1e-13)
        np.testing.assert_allclose((out-out.mean(1,keepdims=True))/out.std(1,keepdims=True),
            (f-f.mean(1,keepdims=True))/f.std(1,keepdims=True),rtol=0,atol=1e-14)
        self.assertFalse(np.allclose(out,target));self.assertTrue(np.all(d['gain']>=0))

    def test_moment_match_is_per_query_not_pooled(self):
        f=np.tile(np.arange(8,dtype=float),(2,1))
        target=np.stack((5+2*f[0],-8+.25*f[1]))
        out,d=global_affine_moments(f,target)
        np.testing.assert_allclose(out,target,rtol=0,atol=1e-14)
        np.testing.assert_allclose(d['gain'],[2.,.25],rtol=0,atol=1e-14)

    def test_constant_cases_not_silently_dropped_or_noised(self):
        out,_=global_affine_moments(np.full((1,8),3.),np.full((1,8),7.))
        np.testing.assert_array_equal(out,np.full((1,8),7.))
        with self.assertRaisesRegex(ValueError,'SOURCE_MOMENT_DEGENERATE'):
            global_affine_moments(np.full((1,8),3.),np.arange(8).reshape(1,8))

    def test_zero_event_is_exact_factual_and_kept(self):
        f=np.array([[1.,2.,3.,4.]])
        np.testing.assert_array_equal(impulse_only(f,np.zeros_like(f),.791286332),f)

    def test_shapes_nonfinite_and_invalid_ratio_rejected(self):
        good=np.ones((1,4))
        for ratio in (0.,-1.,np.nan,np.inf):
            with self.subTest(ratio=ratio),self.assertRaises(ValueError): impulse_only(good,good,ratio)
        for a,b in ((good,np.ones((1,3))),(np.array([1.,2.]),np.array([1.,2.])),(good,np.full((1,4),np.nan))):
            with self.assertRaises(ValueError): global_affine_moments(a,b)

    def test_highest_combination_does_not_imply_positive_interaction(self):
        v=factorial(.8,.84,.85,.87)
        self.assertAlmostEqual(v['interaction'],-.02)
        self.assertAlmostEqual(v['timing_given_amplitude'],.02)
        self.assertAlmostEqual(v['amplitude_given_timing'],.03)


if __name__ == '__main__': unittest.main()
