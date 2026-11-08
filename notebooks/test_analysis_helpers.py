"""Small regression checks; these do not reproduce the full historical notebook experiment."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from a2_analysis import eda, isoforest


def test_notebooks_are_valid_and_code_cells_parse():
    import nbformat
    for path in Path(__file__).parent.glob('*A2*.ipynb'):
        notebook = nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
            assert not any(output.get('output_type') == 'error' for output in cell.get('outputs',[]))


def test_negative_missing_codes_and_temporal_fold():
    values = pd.Series([-2.,-1.,0.,2.],name='intended_balcon_amount')
    assert eda.missing_mask(values).tolist() == [True,True,False,False]
    frame = pd.DataFrame({'month':[0,1,2,3,4,5]})
    train, validation = isoforest.split_by_month(frame,range(3),3)
    assert train.month.max() < validation.month.min()


def test_unsupervised_fit_and_metric_diagnostics():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(100,4))
    model = isoforest.fit_isolation_forest(train,n_estimators=5,max_samples=32)
    scores = isoforest.anomaly_score(model,train)
    assert np.isfinite(scores).all()
    y = np.tile([0,1],50)
    result = isoforest.evaluate(y,scores)
    assert 0 <= result['pr_auc'] <= 1
    assert result['achieved_fpr'] <= .05
    expected = np.quantile(scores,.95)
    assert isoforest.threshold_for_rate(scores,.05) == expected


def test_saved_results_are_separate_from_champion():
    root = Path(__file__).parents[1]
    comparison = pd.read_csv(root/'notebooks/a2_outputs/tables/isoforest_comparison.csv')
    assert set(comparison.val_month) == {3,4,5}
    assert len(comparison) == 6
    registry = json.loads((root/'experiment_suite/configs/final_models.json').read_text())
    assert registry['datasets']['baf']['model'] == 'catboost'
