"""Reproducible grouped evaluation. Model selection never consults the test set."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'reports'/'v2'/'.mplconfig'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate

from ml_v2 import FEATURES, make_model

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'processed_v2'
OUT = ROOT / 'modelling' / 'v2'
REPORT = ROOT / 'reports' / 'v2'


def metrics(y, p):
    return {'mae_billion': float(mean_absolute_error(y, p)),
            'rmse_billion': float(np.sqrt(mean_squared_error(y, p))),
            'r2_price': float(r2_score(y, p)),
            'median_absolute_error_billion': float(np.median(np.abs(y-p)))}


def load(split):
    x = pd.read_csv(DATA / f'X_{split}.csv')
    y = pd.read_csv(DATA / f'y_{split}.csv')
    assert x.record_id.equals(y.record_id)
    return x.record_id, x[FEATURES], y.price


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    ids, x, y = load('train')
    manifest = pd.read_csv(DATA/'split_manifest.csv').set_index('record_id')
    groups = manifest.loc[ids, 'candidate_group_id'].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    fold_rows = []
    for k, (a,b) in enumerate(folds):
        assert not set(groups[a]).intersection(groups[b])
        fold_rows.extend({'record_id': ids.iloc[i], 'validation_fold': k} for i in b)
    pd.DataFrame(fold_rows).to_csv(REPORT/'cv_folds.csv', index=False)
    # EDA used for development is confined to training rows.
    x.assign(price=y).describe(include='all').to_csv(REPORT/'train_summary.csv')
    x.isna().mean().rename('missing_fraction').to_csv(REPORT/'train_missingness.csv')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(y, bins=60, color='#277c74'); axes[0].set(xlabel='Asking price (billion VND)',ylabel='Train listings')
    axes[1].hist(np.log1p(y), bins=40,color='#bb6259');axes[1].set(xlabel='log1p(price)',ylabel='Train listings')
    fig.tight_layout();fig.savefig(REPORT/'train_price_distribution.png',dpi=150);plt.close(fig)
    scoring = {'mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','r2':'r2'}
    baseline = DummyRegressor(strategy='median')
    cv_base = cross_validate(baseline, np.zeros((len(x),1)), y, cv=folds, scoring=scoring)
    baseline.fit(np.zeros((len(x),1)), y)
    records = [{'model':'Baseline median','cv_mae':float(-cv_base['test_mae'].mean()),
                'cv_mae_std':float(cv_base['test_mae'].std()),'cv_rmse':float(-cv_base['test_rmse'].mean())}]
    models = {'Baseline median':baseline}
    configs = [
        ('Ridge', make_model(Ridge(solver='lsqr'), scale=True),
         {'regressor__model__alpha':[1.,10.,100.,1000.]}),
        ('Random Forest', make_model(RandomForestRegressor(n_estimators=160, max_depth=24, random_state=42, n_jobs=2)),
         {'regressor__model__min_samples_leaf':[2,8], 'regressor__model__max_features':[.7,1.]})]
    params = {}
    for name, estimator, grid in configs:
        print(f'Training {name}: {len(folds)} grouped folds',flush=True)
        search = GridSearchCV(estimator, grid, cv=folds, scoring=scoring, refit='mae', n_jobs=1, error_score='raise')
        search.fit(x,y)
        pd.DataFrame(search.cv_results_).to_csv(REPORT/f'{name.replace(" ","_")}_cv.csv',index=False)
        i = search.best_index_
        records.append({'model':name,'cv_mae':float(-search.cv_results_['mean_test_mae'][i]),
                        'cv_mae_std':float(search.cv_results_['std_test_mae'][i]),
                        'cv_rmse':float(-search.cv_results_['mean_test_rmse'][i])})
        params[name] = search.best_params_
        models[name] = search.best_estimator_
        print(records[-1],flush=True)
    selected = min(records, key=lambda row:row['cv_mae'])['model']
    # Freeze selection before accessing holdout values.
    (OUT/'selection.json').write_text(json.dumps({'selected':selected,'criterion':'minimum mean grouped CV MAE, original price scale','params':params},indent=2),encoding='utf-8')
    test_ids, xt, yt = load('test')
    assert not set(groups).intersection(set(manifest.loc[test_ids,'candidate_group_id']))
    predictions = pd.DataFrame({'record_id':test_ids,'actual_price':yt})
    for row in records:
        name = row['model']; est = models[name]
        p = est.predict(np.zeros((len(xt),1)) if name=='Baseline median' else xt)
        assert np.isfinite(p).all()
        row.update(metrics(yt,p));predictions[name] = p
    artifact = {'model':models[selected], 'name':selected,'features':FEATURES,
                'target':'asking_price_billion_vnd','returns_original_price':True,
                'baseline':selected=='Baseline median', 'params':params.get(selected,{}),
                'versions':{k:importlib.metadata.version(k) for k in ['pandas','numpy','scikit-learn','joblib','streamlit','plotly','matplotlib']},
                'dataset_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')}}
    joblib.dump(artifact,OUT/'model.joblib')
    loaded = joblib.load(OUT/'model.joblib')
    probe = np.zeros((5,1)) if artifact['baseline'] else xt.iloc[:5]
    np.testing.assert_allclose(loaded['model'].predict(probe),models[selected].predict(probe))
    pd.DataFrame(records).to_csv(REPORT/'metrics.csv',index=False)
    predictions.to_csv(REPORT/'test_predictions.csv',index=False)
    detail = xt.copy();detail['actual']=yt;detail['predicted']=predictions[selected]
    detail['source']=manifest.loc[test_ids,'source'].to_numpy()
    detail['price_band']=pd.cut(yt,[-np.inf,3,10,30,np.inf],labels=['<3','3-10','10-30','>30'],right=False)
    segments=[]
    for column in ['source','property_type','city_province','price_band']:
        for label,sub in detail.groupby(column,dropna=False,observed=True):
            vals=metrics(sub.actual,sub.predicted) if len(sub)>1 else {'mae_billion':float(abs(sub.actual.iloc[0]-sub.predicted.iloc[0])),'rmse_billion':float(abs(sub.actual.iloc[0]-sub.predicted.iloc[0])),'r2_price':None}
            segments.append({'dimension':column,'segment':str(label),'count':len(sub),**vals})
    pd.DataFrame(segments).to_csv(REPORT/'segment_errors.csv',index=False)
    detail.insert(0,'record_id',test_ids)
    detail['absolute_error']=(detail.actual-detail.predicted).abs()
    detail.nlargest(30,'absolute_error').to_csv(REPORT/'largest_errors.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].scatter(yt,predictions[selected],s=8,alpha=.4,color='#277c74')
    limit=max(yt.max(),predictions[selected].max());axes[0].plot([0,limit],[0,limit],color='#bb6259')
    axes[0].set(xlabel='Actual (billion VND)',ylabel='Predicted (billion VND)')
    axes[1].scatter(predictions[selected],yt-predictions[selected],s=8,alpha=.4,color='#277c74');axes[1].axhline(0,color='#bb6259')
    axes[1].set(xlabel='Predicted (billion VND)',ylabel='Residual (billion VND)')
    fig.tight_layout();fig.savefig(REPORT/'test_diagnostics.png',dpi=150);plt.close(fig)
    (OUT/'metadata.json').write_text(json.dumps({k:v for k,v in artifact.items() if k!='model'},ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'requirements_v2.lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in artifact['versions'].items())+'\n',encoding='utf-8')
    print('Selected:',selected,flush=True);print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
