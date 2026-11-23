"""Test CSV writer without importing Selenium or accessing the website."""
import ast
import csv
import os
from pathlib import Path
import tempfile
import unittest


class CsvWriterTests(unittest.TestCase):
    def test_existing_header_order(self):
        path=Path(__file__).resolve().parents[1]/'data_scraping/scraper.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['write_rows','ensure_outdir']]
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'sample.csv'
            target.write_text('property_type,price\nHouse,5\n',encoding='utf-8')
            env={'os':os,'csv':csv,'FIELDS':['price','property_type'],'OUTPUT_CSV':str(target)}
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
            env['write_rows']([{'price':8,'property_type':'Apartment'}])
            with target.open(encoding='utf-8') as file:
                records=list(csv.DictReader(file))
            self.assertEqual(records[-1],{'property_type':'Apartment','price':'8'})
            env['FIELDS']=['price','wrong']
            with self.assertRaises(ValueError):
                env['write_rows']([{'price':9,'wrong':'x'}])


if __name__=='__main__':
    unittest.main()
