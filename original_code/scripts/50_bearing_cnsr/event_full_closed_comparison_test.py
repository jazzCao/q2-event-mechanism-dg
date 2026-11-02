"""Known-outcome arithmetic and adversarial-role/chronology tests, no payloads."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import event_full_closed_comparison as c


def fixture():
    cases={};rows=[];methods=('current/Q2','archived_recipe/B0')
    for p,f,target,value in (('A','a','X',.8),('B','b1','X',.2),('B','b2','X',.2),('B','b3','Y',.2)):
        for seed in (0,1,2):
            cases[p+'/'+f+'/s'+str(seed)]=dict(task=dict(protocol=p,fold=f,target=target),seed=seed)
            for a in methods:
                for role in ('last','selected'):
                    n=round(value*10) if role=='last' else 10
                    m=dict(acc=n/10,macro_f1=n/10,recording_ba=n/10,windows=20,recordings=20,
                        window_confusion=[[n,10-n],[10-n,n]],recording_confusion=[[n,10-n],[10-n,n]])
                    rows.append(dict(protocol=p,fold=f,seed=seed,arm=a,role=role,metrics=m))
    return cases,rows,methods


class ComparisonTests(unittest.TestCase):
    def test_dataset_and_task_weight_are_distinct(self):
        cases,rows,methods=fixture();index=c.index_rows(rows,cases,methods)
        v=c.summarize(index,cases,methods,'last')['current/Q2']
        self.assertAlmostEqual(v['dataset_equal']['acc'],.5)
        self.assertAlmostEqual(v['task_equal']['acc'],.35)

    def test_policies_remain_separate(self):
        cases,rows,methods=fixture();index=c.index_rows(rows,cases,methods)
        self.assertAlmostEqual(c.summarize(index,cases,methods,'last')['current/Q2']['dataset_equal']['acc'],.5)
        self.assertEqual(c.summarize(index,cases,methods,'selected')['current/Q2']['dataset_equal']['acc'],1.)

    def test_duplicate_role_rejected(self):
        cases,rows,methods=fixture()
        with self.assertRaises(ValueError):c.index_rows(rows+[copy.deepcopy(rows[0])],cases,methods)

    def test_missing_role_rejected(self):
        cases,rows,methods=fixture()
        with self.assertRaises(ValueError):c.index_rows(rows[:-1],cases,methods)

    def test_undeclared_seed_rejected(self):
        cases,rows,methods=fixture();rows[0]['seed']=9
        with self.assertRaises(ValueError):c.index_rows(rows,cases,methods)

    def test_target_removal_groups_both_source_scenarios(self):
        cases,_,_=fixture();remaining=c.without_target(cases,'B','X')
        self.assertEqual(len(remaining),6)
        self.assertEqual({(v['task']['protocol'],v['task']['fold']) for v in remaining.values()},{('A','a'),('B','b3')})

    def test_declared_absent_class_changes_macro_f1_not_recording_ba(self):
        m=dict(acc=1.,macro_f1=.5,recording_ba=1.,window_confusion=[[3,0],[0,0]],recording_confusion=[[1,0],[0,0]],windows=3,recordings=1)
        c.validate_metrics(m);m['macro_f1']=1.
        with self.assertRaises(ValueError):c.validate_metrics(m)

    def test_parent_not_closed_blocks_before_any_result_read(self):
        with patch.object(Path,'is_file',return_value=False),patch.object(c,'read',side_effect=AssertionError('premature read')):
            with self.assertRaises(ValueError):c.require_closed()


if __name__=='__main__':unittest.main()
