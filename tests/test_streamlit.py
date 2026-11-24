from pathlib import Path
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class StreamlitTests(unittest.TestCase):
    def test_single_batch_and_empty_filters(self):
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=60).run()
        self.assertFalse(app.exception, str(app.exception))
        submit=next(b for b in app.button if b.label=='Ước lượng giá')
        submit.click().run()
        self.assertFalse(app.exception,str(app.exception))
        self.assertTrue(app.success)
        # Use one city for a fast batch test; download data must persist on rerun.
        app.sidebar.multiselect[0].set_value(['Hà Nội']).run()
        next(b for b in app.button if b.label=='Dự đoán các tin đang lọc').click().run()
        self.assertFalse(app.exception,str(app.exception))
        self.assertGreater(len(app.session_state['batch']),0)
        app.run()
        self.assertGreater(len(app.session_state['batch']),0)
        data=pd.read_csv(ROOT/'data/processed/canonical_all.csv',low_memory=False)
        all_types=set(data.property_type.dropna())
        for city, rows in data.groupby('city_province'):
            absent=all_types-set(rows.property_type)
            if absent:
                app.sidebar.multiselect[0].set_value([city])
                app.sidebar.multiselect[1].set_value([sorted(absent)[0]]).run()
                break
        self.assertFalse(app.exception,str(app.exception))
        self.assertEqual(app.metric[0].value,'0')


if __name__=='__main__':
    unittest.main()
