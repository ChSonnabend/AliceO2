import csv
import gzip
import tempfile
import unittest
from pathlib import Path
from filterTransportRoots import filter_roots
from analyzeTransportShadow import read_score_csv
import numpy as np

HEADER = ['csv_schema_version','model','event_id','model_evaluated','valid_score','score',
          'can_avoid_subtree','is_simulation_root','is_hitless_tree_root','shadow_mode',
          'configured_threshold','would_reject','n_descendants']

class PrimaryCsvTest(unittest.TestCase):
    def fixture(self, path):
        rows=[]
        for event in range(4):
            rows.extend([
                ['1','model,"quoted"',str(event),'1','1','0.10000000149011612','0','1','0','1','0.8','0','3'],
                ['1','model,"quoted"',str(event),'1','1','0.89999997615814209','1','1','1','1','0.8','1','2'],
                ['1','model,"quoted"',str(event),'0','0','-1','1','1','1','1','0.8','-1','0'],
                ['1','model,"quoted"',str(event),'1','1','0.89999997615814209','1','0','0','1','0.8','1','0']])
        with path.open('w',newline='') as out:
            writer=csv.writer(out);writer.writerow(HEADER);writer.writerows(rows)
        return rows

    def test_lossless_stream_and_analysis_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'full.csv';out=Path(d)/'roots.csv.gz';rows=self.fixture(source)
            result=filter_roots(source,out,progress_every=0)
            self.assertEqual(result['scanned_rows'],16);self.assertEqual(result['root_rows'],12)
            with gzip.open(out,'rt',newline='') as stream:
                reader=csv.reader(stream);self.assertEqual(next(reader),HEADER)
                self.assertEqual(list(reader),[r for r in rows if r[7]=='1'])
            full=read_score_csv([source],'roots',chunksize=3)
            filtered=read_score_csv([out],'roots',chunksize=2)
            for a,b in zip(full[:3],filtered[:3]):
                np.testing.assert_array_equal(np.concatenate(a),np.concatenate(b))
            self.assertEqual(full[4]['configured_cut_counts'],filtered[4]['configured_cut_counts'])
            with self.assertRaises(FileExistsError):filter_roots(source,out)
            with self.assertRaises(ValueError):filter_roots(source,source)

    def test_bounded_scan_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'full.csv';out=Path(d)/'roots.csv';self.fixture(source)
            result=filter_roots(source,out,max_rows=2,progress_every=0)
            self.assertEqual(result['scanned_rows'],2);self.assertEqual(result['root_rows'],2)
            result=read_score_csv([source],'roots',chunksize=1,max_rows=2)
            self.assertEqual(result[4]['csv_rows'],2)
            with source.open('a') as f:f.write('1,truncated\n')
            bad=Path(d)/'bad.csv.gz'
            with self.assertRaises(ValueError):filter_roots(source,bad,progress_every=0)
            self.assertFalse(bad.exists());self.assertFalse(list(Path(d).glob('*.partial')))

    def test_cache_reanalysis_without_original(self):
        from primaryPruningCache import save_cache, load_cache
        from analyzeTransportShadow import main
        from plotPrimaryPruningQA import run_qa
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'full.csv';self.fixture(source)
            features={}
            truths,scores,groups,keys,details=read_score_csv([source],'roots',chunksize=3,features=features)
            y,s,g=(np.concatenate(parts) for parts in (truths,scores,groups))
            features={name:np.concatenate(parts) for name,parts in features.items()}
            np.testing.assert_array_equal(features['n_descendants'],[3,2]*4)
            features['pdg']=np.array([211,22]*4,dtype=np.float64)
            features['px']=np.ones(8);features['py']=np.zeros(8);features['pz']=np.ones(8)
            cache=root/'selected_tracks.npz'
            meta=dict(version=1,scope='roots',event_keys=keys,details=details,
                      max_csv_rows_per_file=16,partial_input_smoke_test=True)
            save_cache(cache,y,s,g,meta,features)
            source.unlink()
            main(['--cache',str(cache),'--scope','roots','--out',str(root/'qa')])
            report=json.loads((root/'qa/summary.json').read_text())
            self.assertEqual(report['scored_tracks'],8)
            self.assertTrue(report['partial_input_smoke_test'])
            self.assertEqual(report['configured_operating_point']['wrongly_rejected'],0)
            self.assertEqual(report['configured_operating_point']['correctly_rejected'],4)
            qa=run_qa(cache,root/'extra',.8)
            self.assertEqual(qa['rows'],8)
            with (root/'extra/by_pdg.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(sum(int(row['rows']) for row in rows),8)
            self.assertEqual(sum(int(row['correctly_rejected']) for row in rows),4)
            self.assertTrue((root/'extra/by_pt_GeV.png').exists())
            new_y,new_s,new_g,new_meta,new_features=load_cache(root/'qa/selected_tracks.npz')
            np.testing.assert_array_equal(new_y,y);np.testing.assert_array_equal(new_s,s)
            np.testing.assert_array_equal(new_g,g);self.assertEqual(new_meta,meta)

if __name__=='__main__':unittest.main()
