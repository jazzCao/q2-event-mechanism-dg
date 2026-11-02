"""Stdlib metadata-only tests before any numerical rendering or payload read."""
import math
import sys
import unittest
import continuous_event_plan as p


class Tests(unittest.TestCase):
    def test_complete_sources(self):
        units=p.metadata();self.assertEqual(len(units),3);self.assertEqual(sum(len(v['bindings']) for v in units),9)
        self.assertTrue(all(v['case_key'].endswith('/s2') for v in units))

    def test_full_costs(self):
        f=p.forecast(p.metadata())
        for k,n in dict(renderer_calls=316,mode_sample_updates=180580608,filter_calls=8568,legacy_renderer_calls=114,
            direct_renderer_calls=349,event_grid_scans=172291584,
            legacy_fft_1d_transforms=5586,legacy_fft_points=271277568,frontend_fft_1d_transforms=135,frontend_fft_points=650385,
            source_payload_reads=9,source_payload_bytes=42909444,source_hashes=9,owned_writes=35,owned_reads=35,owned_hashes=70).items():self.assertEqual(f[k],n,k)

    def test_nonrational_nodes_and_moments(self):
        q=p.ratios();self.assertEqual(len(q),7);self.assertEqual([v['value'] for v in q[:3]],[2/3,1.,1.5])
        self.assertTrue(all(2/3<v['value']<1.5 for v in q[3:]))
        x=[math.log(v['value'])/math.log(1.5) for v in q[3:]]
        weights=[(18-math.sqrt(30))/36,(18+math.sqrt(30))/36,(18+math.sqrt(30))/36,(18-math.sqrt(30))/36]
        self.assertAlmostEqual(sum(weights),2.);self.assertAlmostEqual(sum(w*z*z for w,z in zip(weights,x)),2/3)
        self.assertAlmostEqual(sum(w*z**7 for w,z in zip(weights,x)),0.)

    def test_zero_neural_import_and_fixed_full_sizes(self):
        self.assertNotIn('torch',sys.modules);spec=p.fixtures();self.assertEqual(len(spec),32)
        self.assertEqual(sum(v['n']==32768 and v['channels']==48 for v in spec),1)
        self.assertTrue(all(v['ratios'] in (7,9) for v in spec))

    def test_every_source_query_has_original_observation_support(self):
        from continuous_event_core import reference_interval
        for unit in p.metadata():
            for ratio in (2/3,1.,1.5):
                lo,hi=reference_interval(unit['n'],unit['size'],ratio)
                for start in unit['group']['starts']:
                    self.assertGreaterEqual(start,lo)
                    self.assertLessEqual(start+unit['group']['window'],hi)
