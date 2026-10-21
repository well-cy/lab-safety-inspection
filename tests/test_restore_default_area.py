"""BUG-010: isolated API and concurrent restoration regression."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
tmp = tempfile.TemporaryDirectory()
for key, leaf in [('DB_PATH','db.sqlite'),('OUTPUTS_DIR','outputs'),('UPLOAD_DIR','uploads')]:
    os.environ['LABSAFETY_'+key] = str(Path(tmp.name)/leaf)
from backend import database as db
from backend.main import app
from fastapi.testclient import TestClient

class RestoreTests(unittest.TestCase):
    def setUp(self):
        db.init_db()
        self.lab = db.create_lab('test-'+os.urandom(8).hex())
        self.client = TestClient(app)
    def request(self, lab=None):
        return self.client.post(f'/api/settings/lab/{self.lab if lab is None else lab}/restore-default-area')
    def test_restore_after_delete_and_restart(self):
        first=self.request().json()
        db.delete_area(first['area_id'])
        db.init_db()
        self.assertEqual(db.load_areas(self.lab), [])
        r=self.request(); self.assertEqual(r.status_code,200)
        area=next(x for x in db.list_labs_with_areas() if x['id']==self.lab)['areas'][0]
        self.assertEqual([area[k] for k in ('x1','y1','x2','y2')],[.22,.18,.88,.92])
        self.assertEqual(area['required_ppe'],['mask','gloves','lab_coat'])
    def test_repeat(self):
        a=self.request().json();b=self.request().json()
        self.assertEqual(a['area_id'],b['area_id']);self.assertFalse(b['created'])
    def test_concurrent(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            results=list(pool.map(lambda _:db.restore_default_area(self.lab),range(12)))
        self.assertEqual(sum(r['created'] for r in results),1)
        self.assertEqual(len({r['area_id'] for r in results}),1)
    def test_existing_custom_untouched(self):
        db.create_area(self.lab,'custom',0,0,1,1,['goggles'])
        before=db.list_labs_with_areas()
        self.assertEqual(self.request().status_code,409)
        self.assertEqual(db.list_labs_with_areas(),before)
    def test_modified_default_untouched(self):
        area=self.request().json()['area_id']
        with db.get_conn() as c:c.execute('UPDATE safety_rules SET required=0 WHERE area_id=?',(area,))
        self.assertEqual(self.request().status_code,409)
        with db.get_conn() as c:self.assertEqual(c.execute('SELECT SUM(required) FROM safety_rules WHERE area_id=?',(area,)).fetchone()[0],0)
    def test_missing(self):
        self.assertEqual(self.request(999999).status_code,404)
    def test_history_and_other_labs_unchanged(self):
        rid=db.save_detection_record(self.lab,'image','history',1,0,1,1)
        before=next(x for x in db.list_labs_with_areas() if x['id']==1)
        self.assertEqual(self.request().status_code,200)
        self.assertEqual(next(x for x in db.list_labs_with_areas() if x['id']==1),before)
        with db.get_conn() as c:
            self.assertEqual(c.execute('SELECT source_name FROM detection_records WHERE id=?',(rid,)).fetchone()[0],'history')
            self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(),[])

if __name__=='__main__':unittest.main()
