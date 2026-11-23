"""Rebuild auditable datasets from raw CSVs; no learned imputation or scaling."""
from pathlib import Path
import hashlib
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'processed_v2'
FEATURES = ['area', 'front_width', 'n_bedrooms', 'n_bathrooms', 'n_floors',
            'city_province', 'district', 'property_type', 'legal', 'interior']


def clean(value):
    if pd.isna(value):
        return None
    value = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(value))).strip()
    return value if value.lower() not in {'', 'nan', 'none', 'null', 'not defined'} else None


def key(value):
    value = clean(value)
    if value is None:
        return ''
    value = unicodedata.normalize('NFD', value.lower().replace('đ', 'd'))
    return ''.join(c for c in value if unicodedata.category(c) != 'Mn')


def number(value):
    """Parse one scalar with Vietnamese separators; reject ambiguous/range text."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value) if math.isfinite(float(value)) else np.nan
    s = clean(value)
    m = re.fullmatch(r'([+-]?\d[\d.,]*)(?:\s*(?:m²|m2|m|phòng|tầng|wc))?\.?', s, re.I)
    if not m:
        return np.nan
    s = m.group(1)
    if ',' in s:
        s = s.replace('.', '').replace(',', '.')
    elif re.fullmatch(r'\d{1,3}(?:\.\d{3})+', s):
        s = s.replace('.', '')
    try:
        result = float(s)
        return result if math.isfinite(result) else np.nan
    except ValueError:
        return np.nan


def price_bds(value, area):
    s = key(value)
    if not s:
        return np.nan, 'missing'
    if 'thoa thuan' in s:
        return np.nan, 'negotiable'
    m = re.fullmatch(r'([\d.,]+)\s*(ty|trieu|nghin|dong)(?:\s*/\s*m[²2])?', s)
    if not m:
        return np.nan, 'unparsed'
    price = number(m.group(1)) * {'ty': 1, 'trieu': .001, 'nghin': .000001, 'dong': 1e-9}[m.group(2)]
    if '/' in s:
        if pd.isna(area) or area <= 0:
            return np.nan, 'unit_price_without_area'
        return price * area, 'derived_from_unit_price'
    return price, 'parsed_total'


def city(value):
    s = clean(value)
    if s is None:
        return None
    aliases = {'tp.hcm', 'tphcm', 'tp hcm', 'ho chi minh', 'sai gon',
               'tp. ho chi minh', 'tp ho chi minh', 'thanh pho ho chi minh'}
    if key(s) in aliases:
        return 'Hồ Chí Minh'
    if key(s) in {'ha noi', 'tp. ha noi', 'thanh pho ha noi'}:
        return 'Hà Nội'
    if re.sub(r'[^a-z0-9 ]', '', key(s)).replace('  ', ' ') == 'ba ria vung tau':
        return 'Bà Rịa - Vũng Tàu'
    return s


def district(value):
    s = clean(value)
    if s is None:
        return None
    # Keep administrative names as recorded historically; only align prefixes.
    return re.sub(r'^(?:quận|huyện|thành phố|thị xã|tp\.?|tx\.?)\s+', '', s, flags=re.I).strip()


TYPE_MAP = {
    'nha rieng': 'Nhà riêng', 'nha mat pho': 'Nhà mặt phố',
    'nha mat tien': 'Nhà mặt phố', 'mat tien': 'Nhà mặt phố',
    'nha hem ngo': 'Nhà hẻm/ngõ', 'nha hem, ngo': 'Nhà hẻm/ngõ',
    'can ho chung cu': 'Chung cư', 'biet thu': 'Biệt thự',
    'biet thu, villa': 'Biệt thự', 'nha biet thu, lien ke': 'Biệt thự/liền kề',
}


def legal(value):
    s = key(value)
    if not s:
        return 'Chưa rõ'
    if 'cho so' in s or 'dang lam so' in s:
        return 'Đang chờ giấy chứng nhận'
    if 'vi bang' in s:
        return 'Vi bằng'
    if 'hop dong' in s or 'hdmb' in s:
        return 'Hợp đồng'
    if 'so chung' in s:
        return 'Sổ chung'
    if 'so rieng' in s or 'so hong rieng' in s or 'so do rieng' in s:
        return 'Sổ riêng'
    if 'khong' in s or 'chua' in s:
        return 'Khác/chưa phân loại'
    if 'so do' in s or 'so hong' in s or s.rstrip('.') == 'co so':
        return 'Giấy chứng nhận; chưa rõ riêng/chung'
    if s == 'giay to hop le':
        return 'Giấy tờ hợp lệ theo tin đăng'
    return 'Khác/chưa phân loại'


def interior(value):
    s = key(value)
    if not s:
        return 'Chưa rõ'
    if 'khong noi that' in s or s.rstrip('.') == 'khong':
        return 'Không nội thất'
    if 'cao cap' in s:
        return 'Cao cấp'
    if 'day du' in s or 'full' in s:
        return 'Đầy đủ'
    if 'co ban' in s:
        return 'Cơ bản'
    return 'Khác/chưa phân loại'


def digest(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def write(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def build_source(filename, source):
    p = ROOT / 'data' / 'raw' / filename
    raw = pd.read_csv(p)
    original_cols = list(raw.columns)
    duplicate = raw.duplicated(keep='first')
    raw['record_id'] = [f'{source}:{i+1:06d}' for i in range(len(raw))]
    raw['source_row_id'] = np.arange(1, len(raw)+1)
    fingerprints = pd.util.hash_pandas_object(raw[original_cols], index=False)
    first = dict(zip(fingerprints[~duplicate], raw.loc[~duplicate, 'record_id']))
    removed = raw.loc[duplicate, ['record_id', 'source_row_id']].copy()
    removed['kept_record_id'] = fingerprints[duplicate].map(first)
    removed['source'] = source
    removed['reason'] = 'exact_duplicate_all_original_columns'
    d = raw.loc[~duplicate].copy()
    out = pd.DataFrame(index=d.index)
    out['record_id'] = d.record_id
    out['source_row_id'] = d.source_row_id
    out['source'] = source
    # Preserve every original field, not just the fields selected for modeling.
    for col in original_cols:
        out['raw__' + col] = d[col]

    def get(col):
        return d[col] if col in d else pd.Series(None, index=d.index, dtype='object')

    out['url'] = get('url').map(clean)
    out['title'] = get('title').map(clean)
    out['date_of_posting_raw'] = get('date_of_posting')
    out['collected_at'] = None
    out['address'] = get('location_details' if source == 'batdongsan' else 'address').map(clean)
    if source == 'batdongsan':
        parts = get('location').map(clean).map(lambda s: s.split(',') if isinstance(s, str) else [])
        out['city_province_raw'] = parts.map(lambda v: v[-1].strip() if len(v) >= 2 else None)
        out['district_raw'] = parts.map(lambda v: v[-2].strip() if len(v) >= 2 else None)
    else:
        out['city_province_raw'] = get('city_province')
        out['district_raw'] = get('district')
    out['city_province'] = out.city_province_raw.map(city)
    out['district'] = out.district_raw.map(district)
    out['property_type'] = get('property_type').map(lambda s: TYPE_MAP.get(key(s), 'Khác/chưa phân loại'))
    out['area'] = get('area').map(number)
    for col in ['front_width', 'front_road_width', 'n_bedrooms', 'n_bathrooms', 'n_floors']:
        out[col] = get(col).map(number)
    issues = []
    for col in ['area', 'front_width', 'front_road_width', 'n_bedrooms', 'n_bathrooms', 'n_floors']:
        unparsed = get(col).notna() & out[col].isna()
        invalid = out[col].notna() & (out[col] < 0)
        if col in ['area', 'front_width', 'front_road_width']:
            invalid |= out[col].eq(0)
        if col.startswith('n_'):
            invalid |= out[col].notna() & out[col].mod(1).ne(0)
        for idx in out.index[unparsed | invalid]:
            issues.append({'record_id': out.at[idx, 'record_id'], 'field': col,
                           'raw_value': get(col).loc[idx], 'reason': 'unparsed' if unparsed.loc[idx] else 'invalid_value'})
        out.loc[invalid, col] = np.nan
    if source == 'batdongsan':
        parsed = [price_bds(v, a) for v, a in zip(get('price'), out.area)]
        out['price'] = [v[0] for v in parsed]
        out['price_status'] = [v[1] for v in parsed]
    else:
        # Existing provider numeric prices are interpreted as billion VND, as in
        # the reference project. The original collection provenance is unknown.
        out['price'] = pd.to_numeric(get('price'), errors='coerce')
        out['price_status'] = np.where(out.price.notna(), 'provider_numeric_billion_assumed', 'missing')
        units = get('area_unit').map(key)
        bad_units = units.ne('m²') & units.ne('m2')
        for idx in out.index[bad_units]:
            issues.append({'record_id': out.at[idx, 'record_id'], 'field': 'area',
                           'raw_value': get('area_unit').loc[idx], 'reason': 'unverified_area_unit'})
        out.loc[bad_units, 'area'] = np.nan
    invalid_price = out.price.notna() & (~np.isfinite(out.price) | out.price.le(0))
    out.loc[invalid_price, 'price_status'] = 'invalid_nonpositive_or_nonfinite'
    out.loc[invalid_price, 'price'] = np.nan
    for idx in out.index[out.price_status.isin(['unparsed', 'unit_price_without_area', 'invalid_nonpositive_or_nonfinite'])]:
        issues.append({'record_id': out.at[idx, 'record_id'], 'field': 'price',
                       'raw_value': get('price').loc[idx], 'reason': out.at[idx, 'price_status']})
    out['legal'] = get('legal' if source == 'batdongsan' else 'legal_docs').map(legal)
    out['interior'] = get('interior').map(interior)
    out['latitude'] = pd.to_numeric(get('latitude'), errors='coerce')
    out['longitude'] = pd.to_numeric(get('longitude'), errors='coerce')
    valid_coords = out.latitude.between(-90, 90) & out.longitude.between(-180, 180)
    out['coordinate_status'] = np.where(valid_coords, 'source_coordinates_unverified', 'missing_or_invalid')
    out.loc[~valid_coords, ['latitude', 'longitude']] = np.nan
    out['has_interior_info'] = get('interior').map(clean).notna()
    out['address_key'] = out.address.map(key).str.replace(r'[^\w\s,]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    # Review candidates, not an automated outlier-removal rule.
    out['review_extreme'] = out.price.gt(500) | out.area.lt(10) | out.area.gt(1500)
    info = {'input_rows': len(raw), 'exact_duplicates_removed': int(duplicate.sum()),
            'remaining_rows': len(out), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    return out.reset_index(drop=True), removed, issues, info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b, bd, bi, bs = build_source('batdongsan_com_vn.csv', 'batdongsan')
    m, md, mi, ms = build_source('muaban_net.csv', 'muaban')
    d = pd.concat([b, m], ignore_index=True)
    repeated_url = d.url.notna() & d.duplicated(['source', 'url'], keep=False)
    # Exact copies already removed. Remaining repeated URLs are conflicting
    # snapshots with no collection timestamp: quarantine from supervised data.
    d['url_conflict'] = repeated_url
    d['eligible_for_model'] = d.price.gt(0) & d.area.gt(0) & ~d.url_conflict
    d['exclusion_reason'] = [
        ';'.join(reason for flag, reason in [(pd.isna(p), 'missing_price'),
                                             (pd.isna(a), 'missing_area'),
                                             (u, 'conflicting_url')] if flag)
        for p, a, u in zip(d.price, d.area, d.url_conflict)]
    match_cols = ['city_province', 'district', 'address_key', 'property_type',
                  'area', 'n_bedrooms', 'n_bathrooms', 'n_floors']
    # No target-derived information in the group key. Same-address matching is
    # deliberately conservative for split isolation, never proof of identity.
    keys = []
    for row in d.to_dict('records'):
        if row['address_key'] and pd.notna(row['area']):
            keys.append(digest(json.dumps([None if pd.isna(row[c]) else row[c] for c in match_cols], ensure_ascii=False)))
        else:
            keys.append(digest(row['record_id']))
    d['candidate_group_id'] = keys
    d['candidate_group_size'] = d.groupby('candidate_group_id').record_id.transform('size')
    eligible = d[d.eligible_for_model].copy()
    eligible['split'] = eligible.candidate_group_id.map(
        lambda s: 'test' if int(digest('split42:' + s)[:8], 16) / 2**32 < .2 else 'train')
    # EDA-only target-derived columns are never exported as model inputs.
    d['price_per_m2'] = d.price * 1000 / d.area
    d['log_price'] = np.log1p(d.price)
    write(d, 'canonical_all.csv')
    for source in ['batdongsan', 'muaban']:
        write(d[d.source.eq(source)], f'{source}_canonical.csv')
    write(pd.concat([bd, md], ignore_index=True), 'exact_duplicates_removed.csv')
    write(d[d.url_conflict], 'url_conflicts.csv')
    write(d[d.candidate_group_size.gt(1)], 'duplicate_candidates.csv')
    write(d[~d.eligible_for_model], 'excluded_from_model.csv')
    write(pd.DataFrame(bi + mi, columns=['record_id','field','raw_value','reason']), 'parse_issues.csv')
    write(d[d.review_extreme], 'extreme_values_review.csv')
    for original, normalized in [('city_province_raw','city_province'),
                                 ('district_raw','district'),
                                 ('raw__property_type','property_type')]:
        mapping = d.groupby(['source', original, normalized], dropna=False).size().reset_index(name='rows')
        write(mapping, f'mapping_{normalized}.csv')
    quality = []
    for source, frame in d.groupby('source'):
        for col in FEATURES + ['price','latitude','longitude']:
            unknown = frame[col].isna() | frame[col].eq('Chưa rõ')
            quality.append({'source':source,'column':col,'rows':len(frame),
                            'missing_or_unknown':int(unknown.sum()),
                            'missing_or_unknown_pct':round(float(unknown.mean()*100),2)})
    write(pd.DataFrame(quality), 'missingness_by_source.csv')
    write(eligible[['record_id','source','candidate_group_id','split']], 'split_manifest.csv')
    for split in ['train','test']:
        subset = eligible[eligible.split.eq(split)]
        write(subset[['record_id'] + FEATURES], f'X_{split}.csv')
        write(subset[['record_id','price']], f'y_{split}.csv')
    train_groups = set(eligible.loc[eligible.split.eq('train'), 'candidate_group_id'])
    test_groups = set(eligible.loc[eligible.split.eq('test'), 'candidate_group_id'])
    assert not train_groups.intersection(test_groups)
    assert not d.record_id.duplicated().any()
    assert d.loc[d.raw__property_type.eq('Nhà riêng'), 'property_type'].eq('Nhà riêng').all()
    assert d.loc[d.source.eq('muaban'), 'n_floors'].isna().sum() == m.n_floors.isna().sum()
    stats = {'sources': {'batdongsan': bs, 'muaban': ms},
             'canonical_rows': len(d), 'model_rows': len(eligible),
             'split_counts': eligible.split.value_counts().to_dict(),
             'url_conflict_rows': int(d.url_conflict.sum()),
             'url_conflict_groups': int(d.loc[d.url_conflict,'url'].nunique()),
             'candidate_groups': int(d.loc[d.candidate_group_size.gt(1),'candidate_group_id'].nunique()),
             'parse_issue_count': len(bi+mi), 'extreme_review_rows': int(d.review_extreme.sum()),
             'city_counts': d.city_province.value_counts().to_dict(),
             'property_counts': d.property_type.value_counts().to_dict(),
             'missing_numeric': d[['price','area','front_width','n_bedrooms','n_bathrooms','n_floors']].isna().sum().to_dict(),
             'model_features': FEATURES,
             'limitations': ['Muaban provenance and numeric price unit require provider confirmation.',
                            'No imputation, scaling, external geocoding or model training performed.',
                            'Candidate groups do not establish true property identity; unknown duplicates may remain.',
                            'Historical location labels retained; not a current administrative-boundary migration.',
                            'Test split is approximate 20 percent by deterministic candidate group hash.']}
    (OUT/'quality_report.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
