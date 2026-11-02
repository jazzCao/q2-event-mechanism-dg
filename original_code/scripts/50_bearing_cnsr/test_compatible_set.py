"""Synthetic-only set/Gram/role regressions; CUDA loss equivalence is a separate real P0."""
import unittest
import numpy as np
from compatible_set_core import gram_random, assemble
from source_rate_core import shared_edit_mask
from source_rate_screen_data import check_vectors


class CoreTests(unittest.TestCase):
    def test_full_gram(self):
        rng=np.random.RandomState(7)
        for size in (2,8,2048):
            a=rng.normal(size=(2,size)); r=np.stack(gram_random(*a,'source-test'))
            np.testing.assert_allclose(a@a.T,r@r.T,atol=1e-10,rtol=1e-12)

    def test_zero_and_collinear(self):
        a=np.arange(8,dtype=float)
        for pair in ((a,a),(a,-a),(a,a*2),(a,a*0),(a*0,a),(a*0,a*0)):
            p=np.stack(pair); r=np.stack(gram_random(*pair,'degenerate'))
            np.testing.assert_allclose(p@p.T,r@r.T,atol=1e-10)

    def test_deterministic_nonlabel_identity(self):
        a=np.arange(8,dtype=float); x=gram_random(a,-a,'sourceA'); y=gram_random(a,-a,'sourceA'); z=gram_random(a,-a,'sourceB')
        self.assertTrue(all(np.array_equal(p,q) for p,q in zip(x,y)))
        self.assertFalse(np.array_equal(x[0],z[0]))

    def test_finite_shape_rejection(self):
        for a,b in ((np.ones(1),np.ones(1)),(np.ones(8),np.ones(7)),(np.full(8,np.nan),np.ones(8))):
            with self.assertRaises(ValueError): gram_random(a,b,'bad')

    def test_all_bank_rows_and_alias(self):
        g=dict(group_key=np.array('g'),origin=np.array('r'),starts=np.array([1,2,3]))
        for name in ('clean','fixed','plus','minus','random_plus','random_minus'):
            g[name]=np.zeros((3,2048) if name=='clean' else (3,2,2048),np.float32)
        case=dict(bank=[dict(group='g',record='r',domain='A',label=2)])
        a=assemble(case,{'g':g}); self.assertEqual(len(a['y']),3)
        check_vectors(a,dict(y=[2]*3,g=['r']*3,domain=['A']*3,offset=[65537,65538,65539]))
        self.assertTrue(np.array_equal(a['fixed'],a['plus']))
        case['bank'][0]['record']='wrong'
        with self.assertRaises(ValueError): assemble(case,{'g':g})

    def test_shared_mask_exact_rng_geometry(self):
        for d,k,p in ((3,3,13),(2,8,7),(2,4,15)):
            a=np.random.RandomState(10).uniform(size=d*k*p); mask=shared_edit_mask(a,d,k,p).reshape(d,k,p)
            self.assertTrue(all(np.array_equal(v,mask[0]) for v in mask))

    def test_no_torch_import(self):
        import sys
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__': unittest.main(verbosity=2)
