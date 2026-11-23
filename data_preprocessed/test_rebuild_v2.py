"""Focused regression tests for the corrected data semantics and exported split."""
import unittest
import math
import pandas as pd
from rebuild_v2 import number, price_bds, city, district, legal, interior, TYPE_MAP, key, OUT, FEATURES


class ParsingTests(unittest.TestCase):
    def test_numeric_units(self):
        self.assertEqual(number('75 m²'), 75)
        self.assertEqual(number('1.200 m²'), 1200)
        self.assertEqual(number('1.200,5 m²'), 1200.5)
        self.assertEqual(number('4,5 m'), 4.5)
        self.assertEqual(number(4.5), 4.5)
        self.assertTrue(math.isnan(number('3-4 phòng')))

    def test_price(self):
        self.assertAlmostEqual(price_bds('7,7 tỷ', 100)[0], 7.7)
        self.assertAlmostEqual(price_bds('850 triệu', 100)[0], .85)
        self.assertAlmostEqual(price_bds('80 triệu/m²', 100)[0], 8)
        self.assertEqual(price_bds('Thỏa thuận', 100)[1], 'negotiable')
        self.assertTrue(math.isnan(price_bds('80 triệu/m²', float('nan'))[0]))
        self.assertTrue(math.isnan(price_bds('khoảng 5-6 tỷ', 100)[0]))

    def test_categories(self):
        for name in [' TP.HCM ', 'tphcm', 'Sài Gòn', 'Hồ Chí Minh']:
            self.assertEqual(city(name), 'Hồ Chí Minh')
        self.assertEqual(city('Bà Rịa Vũng Tàu'), city('Bà Rịa - Vũng Tàu'))
        self.assertEqual(district('Quận Bình Thạnh'), 'Bình Thạnh')
        self.assertEqual(TYPE_MAP[key('Nhà riêng')], 'Nhà riêng')
        self.assertNotEqual(legal('Sổ hồng'), 'Sổ riêng')
        self.assertEqual(legal(None), 'Chưa rõ')
        self.assertEqual(interior(None), 'Chưa rõ')
        self.assertEqual(interior('Không nội thất'), 'Không nội thất')


class ExportTests(unittest.TestCase):
    def test_lineage_and_missing(self):
        d = pd.read_csv(OUT/'canonical_all.csv', low_memory=False)
        removed = pd.read_csv(OUT/'exact_duplicates_removed.csv')
        self.assertEqual(len(d)+len(removed), 13828)
        self.assertFalse(d.record_id.duplicated().any())
        self.assertTrue(set(removed.kept_record_id).issubset(set(d.record_id)))
        m = d[d.source.eq('muaban')]
        for col in ['n_bedrooms','n_bathrooms','n_floors']:
            self.assertTrue(m.loc[m['raw__'+col].isna(), col].isna().all())
        self.assertTrue(d.loc[d.raw__property_type.eq('Nhà riêng'), 'property_type'].eq('Nhà riêng').all())

    def test_split_and_features(self):
        manifest = pd.read_csv(OUT/'split_manifest.csv')
        self.assertTrue(manifest.groupby('candidate_group_id').split.nunique().eq(1).all())
        all_ids = []
        for split in ['train','test']:
            x = pd.read_csv(OUT/f'X_{split}.csv')
            y = pd.read_csv(OUT/f'y_{split}.csv')
            self.assertEqual(list(x.columns), ['record_id']+FEATURES)
            self.assertEqual(x.record_id.tolist(), y.record_id.tolist())
            self.assertEqual(set(x.record_id), set(manifest.loc[manifest.split.eq(split),'record_id']))
            all_ids.extend(x.record_id)
        self.assertEqual(len(all_ids),len(set(all_ids)))
        conflicts = pd.read_csv(OUT/'url_conflicts.csv')
        self.assertFalse(set(all_ids).intersection(conflicts.record_id))


if __name__ == '__main__':
    unittest.main()
