from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from ml_pipeline import FEATURES, make_model, prepare_inputs

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_train_only_imputation(self):
        x=pd.read_csv(ROOT/'data/processed/X_train.csv')[FEATURES].head(50)
        y=pd.read_csv(ROOT/'data/processed/y_train.csv').price.head(50)
        x.loc[x.index[:5],'front_width']=np.nan
        model=make_model(Ridge(solver='lsqr',alpha=10),True).fit(x,y)
        imputer=model.regressor_.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
        before=imputer.statistics_.copy()
        unseen=x.iloc[:1].copy();unseen['city_province']='Unseen city';unseen['front_width']=9.9
        self.assertTrue(np.isfinite(model.predict(unseen)).all())
        np.testing.assert_array_equal(before,imputer.statistics_)
        self.assertAlmostEqual(before[1],x.front_width.median())

    def test_input_validation(self):
        x=pd.read_csv(ROOT/'data/processed/X_train.csv')[FEATURES].head(1)
        with self.assertRaises(ValueError):
            prepare_inputs(x.drop(columns='area'))
        x['area']=-1
        with self.assertRaises(ValueError):
            prepare_inputs(x)

    def test_artifact_prediction_parity(self):
        obj=joblib.load(ROOT/'modelling/model.joblib')
        x=pd.read_csv(ROOT/'data/processed/X_test.csv').head(8)
        predictions=pd.read_csv(ROOT/'reports/test_predictions.csv').head(8)
        inputs=np.zeros((len(x),1)) if obj['baseline'] else x[FEATURES]
        batch=obj['model'].predict(inputs)
        np.testing.assert_allclose(batch,predictions[obj['name']])
        first=inputs[:1] if obj['baseline'] else inputs.iloc[:1]
        np.testing.assert_allclose(obj['model'].predict(first)[0],batch[0])
        self.assertTrue(obj['returns_original_price'])


if __name__=='__main__':
    unittest.main()
