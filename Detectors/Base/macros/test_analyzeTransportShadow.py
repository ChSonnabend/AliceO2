#!/usr/bin/env python3
import tempfile, subprocess, sys, json
from pathlib import Path
import unittest
import numpy as np
from analyzeTransportShadow import subtree_masks, rates, operating_curve

class ShadowTruthTest(unittest.TestCase):
    def test_reordered_deep_descendants(self):
        # root0 -> 3 -> 2 -> 1; do not assume parent indices precede children.
        np.testing.assert_array_equal(subtree_masks([-1,2,3,0],[0,4,0,0],1),[4,4,4,4])
    def test_generator_link_is_not_transport_ancestry(self):
        np.testing.assert_array_equal(subtree_masks([-1,0,1],[0,0,2],2),[0,2,2])
    def test_cycles_and_missing_parent_fail(self):
        with self.assertRaises(ValueError): subtree_masks([-1,2,1],[0,0,0],1)
        with self.assertRaises(ValueError): subtree_masks([-1,9],[0,0],1)
    def test_invalid_predictions_stay_in_denominators(self):
        result=rates(np.array([0,0,1,1]),np.array([.9,np.nan,.95,np.nan]),.9)
        self.assertEqual(result['hit_tree_loss_rate'],.5)
        self.assertEqual(result['avoidable_rejection_rate'],.5)
    def test_tied_scores_and_keep_all_threshold(self):
        fpr,tpr,cuts=operating_curve(np.array([0,1,1]),np.array([.5,.5,1.],dtype=np.float32))
        self.assertGreater(cuts[0],1)
        np.testing.assert_allclose(fpr,[0,0,1]);np.testing.assert_allclose(tpr,[0,.5,1])
    def test_root_file_to_independent_roc(self):
        import uproot,awkward as ak
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'shadow_Kine.root';out=Path(d)/'qa'
            names={'MCTrack.mMotherTrackId':'var * int32','MCTrack.mProp':'var * uint32',
                   'MCTrack.mTransportModelScore':'var * float32','MCTrack.mTransportModelShadow':'var * bool',
                   'MCEventHeader.FairMCEventHeader.fNPrim':'int32'}
            with uproot.recreate(path) as root:
                root.mktree('o2sim',names)
                root['o2sim'].extend({
                    'MCTrack.mMotherTrackId':ak.Array([[-1,0,1,0]]*8),
                    'MCTrack.mProp':ak.Array([[0,0,0,128]]*8),
                    'MCTrack.mTransportModelScore':ak.Array([[.1,.9,-1.,-1.]]*8),
                    'MCTrack.mTransportModelShadow':ak.Array([[True,True,False,False]]*8),
                    'MCEventHeader.FairMCEventHeader.fNPrim':np.array([2]*8,dtype=np.int32)})
            subprocess.run([sys.executable,str(Path(__file__).with_name('analyzeTransportShadow.py')),'--input',str(path),'--out',str(out)],check=True,stdout=subprocess.DEVNULL)
            result=json.loads((out/'summary.json').read_text())
            self.assertEqual(result['scored_tracks'],16)
            self.assertEqual(result['test']['hit_tree_loss_rate'],0)
            self.assertEqual(result['test']['avoidable_rejection_rate'],1)
            self.assertTrue((out/'roc.png').exists())
            self.assertFalse(set(map(tuple,result['calibration_events'])) & set(map(tuple,result['test_events'])))

if __name__=='__main__':unittest.main()
