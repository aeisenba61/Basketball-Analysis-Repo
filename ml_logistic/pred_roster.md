

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, cross_validation
from sklearn.cross_validation import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
```

    C:\Users\Cynthia\Anaconda3\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    


```python
os.chdir("C:/Users/Cynthia/Desktop/Bootcamp_Github/Basketball-Analysis-Repo/ml_logistic")
```


```python
orig_data = pd.read_csv("../data/clean/draft_nba.csv")
```


```python
# All features
all_feat = ["draft_All_NBA",
            "draft_All.Star",
            "draft_Pk",
            "draft_Games",
            "draft_Minutes.Played",
            "draft_PTS",
            "draft_Win.Share",
            "draft_VORP",
            "draft_WS_per_game",
            "draft_attend_college",
            "Roster"]
```

## Feature selection


```python
data = orig_data[all_feat]
```


```python
skf = StratifiedKFold(data["Roster"], n_folds = 10)
fold_count = 1

corrs = pd.DataFrame()

# Correlations with "Roster"

for train, test in skf:
    
    # Log
    print(f"\n>>> Running fold {fold_count} <<<\n")
    
    # Training & testing dfs
    train_fold = data.iloc[train]
    test_fold = data.iloc[test]
    
    # Best features
    corr = train_fold.corr()["Roster"][train_fold.corr()["Roster"] < 1]
    corrs = corrs.append(corr)
    corrs = corrs.rename(index = {"Roster": f"Fold {fold_count}"})
    print(corr)
    
    # Increment fold
    fold_count += 1
```

    
    >>> Running fold 1 <<<
    
    draft_All_NBA           0.081221
    draft_All.Star          0.092333
    draft_Pk               -0.476737
    draft_Games             0.425231
    draft_Minutes.Played    0.359540
    draft_PTS               0.322347
    draft_Win.Share         0.258547
    draft_VORP              0.154236
    draft_WS_per_game       0.214060
    draft_attend_college    0.394295
    Name: Roster, dtype: float64
    
    >>> Running fold 2 <<<
    
    draft_All_NBA           0.075172
    draft_All.Star          0.087764
    draft_Pk               -0.472424
    draft_Games             0.423274
    draft_Minutes.Played    0.356813
    draft_PTS               0.322062
    draft_Win.Share         0.262194
    draft_VORP              0.154821
    draft_WS_per_game       0.222859
    draft_attend_college    0.392000
    Name: Roster, dtype: float64
    
    >>> Running fold 3 <<<
    
    draft_All_NBA           0.079252
    draft_All.Star          0.090852
    draft_Pk               -0.473138
    draft_Games             0.412525
    draft_Minutes.Played    0.343076
    draft_PTS               0.313548
    draft_Win.Share         0.255142
    draft_VORP              0.154413
    draft_WS_per_game       0.206054
    draft_attend_college    0.394334
    Name: Roster, dtype: float64
    
    >>> Running fold 4 <<<
    
    draft_All_NBA           0.080739
    draft_All.Star          0.093020
    draft_Pk               -0.475965
    draft_Games             0.411227
    draft_Minutes.Played    0.346426
    draft_PTS               0.311299
    draft_Win.Share         0.257448
    draft_VORP              0.155018
    draft_WS_per_game       0.208127
    draft_attend_college    0.376509
    Name: Roster, dtype: float64
    
    >>> Running fold 5 <<<
    
    draft_All_NBA           0.083803
    draft_All.Star          0.090485
    draft_Pk               -0.480477
    draft_Games             0.410928
    draft_Minutes.Played    0.349561
    draft_PTS               0.314959
    draft_Win.Share         0.260038
    draft_VORP              0.158346
    draft_WS_per_game       0.208149
    draft_attend_college    0.355815
    Name: Roster, dtype: float64
    
    >>> Running fold 6 <<<
    
    draft_All_NBA           0.078666
    draft_All.Star          0.090935
    draft_Pk               -0.481536
    draft_Games             0.412884
    draft_Minutes.Played    0.348459
    draft_PTS               0.313178
    draft_Win.Share         0.256029
    draft_VORP              0.154057
    draft_WS_per_game       0.212840
    draft_attend_college    0.364547
    Name: Roster, dtype: float64
    
    >>> Running fold 7 <<<
    
    draft_All_NBA           0.091895
    draft_All.Star          0.104137
    draft_Pk               -0.484338
    draft_Games             0.406784
    draft_Minutes.Played    0.345066
    draft_PTS               0.312609
    draft_Win.Share         0.256753
    draft_VORP              0.155775
    draft_WS_per_game       0.216417
    draft_attend_college    0.383185
    Name: Roster, dtype: float64
    
    >>> Running fold 8 <<<
    
    draft_All_NBA           0.081421
    draft_All.Star          0.093933
    draft_Pk               -0.491569
    draft_Games             0.422848
    draft_Minutes.Played    0.356401
    draft_PTS               0.320499
    draft_Win.Share         0.261862
    draft_VORP              0.155129
    draft_WS_per_game       0.211037
    draft_attend_college    0.345254
    Name: Roster, dtype: float64
    
    >>> Running fold 9 <<<
    
    draft_All_NBA           0.082639
    draft_All.Star          0.094896
    draft_Pk               -0.475576
    draft_Games             0.424135
    draft_Minutes.Played    0.359760
    draft_PTS               0.323905
    draft_Win.Share         0.264547
    draft_VORP              0.160469
    draft_WS_per_game       0.230046
    draft_attend_college    0.364104
    Name: Roster, dtype: float64
    
    >>> Running fold 10 <<<
    
    draft_All_NBA           0.086542
    draft_All.Star          0.098792
    draft_Pk               -0.482906
    draft_Games             0.435905
    draft_Minutes.Played    0.366800
    draft_PTS               0.330893
    draft_Win.Share         0.272318
    draft_VORP              0.164172
    draft_WS_per_game       0.229397
    draft_attend_college    0.371106
    Name: Roster, dtype: float64
    


```python
# Corrs for each fold
corrs
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>draft_All.Star</th>
      <th>draft_All_NBA</th>
      <th>draft_Games</th>
      <th>draft_Minutes.Played</th>
      <th>draft_PTS</th>
      <th>draft_Pk</th>
      <th>draft_VORP</th>
      <th>draft_WS_per_game</th>
      <th>draft_Win.Share</th>
      <th>draft_attend_college</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fold 1</th>
      <td>0.092333</td>
      <td>0.081221</td>
      <td>0.425231</td>
      <td>0.359540</td>
      <td>0.322347</td>
      <td>-0.476737</td>
      <td>0.154236</td>
      <td>0.214060</td>
      <td>0.258547</td>
      <td>0.394295</td>
    </tr>
    <tr>
      <th>Fold 2</th>
      <td>0.087764</td>
      <td>0.075172</td>
      <td>0.423274</td>
      <td>0.356813</td>
      <td>0.322062</td>
      <td>-0.472424</td>
      <td>0.154821</td>
      <td>0.222859</td>
      <td>0.262194</td>
      <td>0.392000</td>
    </tr>
    <tr>
      <th>Fold 3</th>
      <td>0.090852</td>
      <td>0.079252</td>
      <td>0.412525</td>
      <td>0.343076</td>
      <td>0.313548</td>
      <td>-0.473138</td>
      <td>0.154413</td>
      <td>0.206054</td>
      <td>0.255142</td>
      <td>0.394334</td>
    </tr>
    <tr>
      <th>Fold 4</th>
      <td>0.093020</td>
      <td>0.080739</td>
      <td>0.411227</td>
      <td>0.346426</td>
      <td>0.311299</td>
      <td>-0.475965</td>
      <td>0.155018</td>
      <td>0.208127</td>
      <td>0.257448</td>
      <td>0.376509</td>
    </tr>
    <tr>
      <th>Fold 5</th>
      <td>0.090485</td>
      <td>0.083803</td>
      <td>0.410928</td>
      <td>0.349561</td>
      <td>0.314959</td>
      <td>-0.480477</td>
      <td>0.158346</td>
      <td>0.208149</td>
      <td>0.260038</td>
      <td>0.355815</td>
    </tr>
    <tr>
      <th>Fold 6</th>
      <td>0.090935</td>
      <td>0.078666</td>
      <td>0.412884</td>
      <td>0.348459</td>
      <td>0.313178</td>
      <td>-0.481536</td>
      <td>0.154057</td>
      <td>0.212840</td>
      <td>0.256029</td>
      <td>0.364547</td>
    </tr>
    <tr>
      <th>Fold 7</th>
      <td>0.104137</td>
      <td>0.091895</td>
      <td>0.406784</td>
      <td>0.345066</td>
      <td>0.312609</td>
      <td>-0.484338</td>
      <td>0.155775</td>
      <td>0.216417</td>
      <td>0.256753</td>
      <td>0.383185</td>
    </tr>
    <tr>
      <th>Fold 8</th>
      <td>0.093933</td>
      <td>0.081421</td>
      <td>0.422848</td>
      <td>0.356401</td>
      <td>0.320499</td>
      <td>-0.491569</td>
      <td>0.155129</td>
      <td>0.211037</td>
      <td>0.261862</td>
      <td>0.345254</td>
    </tr>
    <tr>
      <th>Fold 9</th>
      <td>0.094896</td>
      <td>0.082639</td>
      <td>0.424135</td>
      <td>0.359760</td>
      <td>0.323905</td>
      <td>-0.475576</td>
      <td>0.160469</td>
      <td>0.230046</td>
      <td>0.264547</td>
      <td>0.364104</td>
    </tr>
    <tr>
      <th>Fold 10</th>
      <td>0.098792</td>
      <td>0.086542</td>
      <td>0.435905</td>
      <td>0.366800</td>
      <td>0.330893</td>
      <td>-0.482906</td>
      <td>0.164172</td>
      <td>0.229397</td>
      <td>0.272318</td>
      <td>0.371106</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Average correlations
avg_corrs = pd.DataFrame(corrs.mean())
avg_corrs.columns = ["Avg corr"]
avg_corrs["Avg corr (Abs)"] = avg_corrs["Avg corr"].abs()
avg_corrs.sort_values(["Avg corr (Abs)"], ascending = False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg corr</th>
      <th>Avg corr (Abs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>draft_Pk</th>
      <td>-0.479467</td>
      <td>0.479467</td>
    </tr>
    <tr>
      <th>draft_Games</th>
      <td>0.418574</td>
      <td>0.418574</td>
    </tr>
    <tr>
      <th>draft_attend_college</th>
      <td>0.374115</td>
      <td>0.374115</td>
    </tr>
    <tr>
      <th>draft_Minutes.Played</th>
      <td>0.353190</td>
      <td>0.353190</td>
    </tr>
    <tr>
      <th>draft_PTS</th>
      <td>0.318530</td>
      <td>0.318530</td>
    </tr>
    <tr>
      <th>draft_Win.Share</th>
      <td>0.260488</td>
      <td>0.260488</td>
    </tr>
    <tr>
      <th>draft_WS_per_game</th>
      <td>0.215899</td>
      <td>0.215899</td>
    </tr>
    <tr>
      <th>draft_VORP</th>
      <td>0.156643</td>
      <td>0.156643</td>
    </tr>
    <tr>
      <th>draft_All.Star</th>
      <td>0.093715</td>
      <td>0.093715</td>
    </tr>
    <tr>
      <th>draft_All_NBA</th>
      <td>0.082135</td>
      <td>0.082135</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top features - greater than .3 corr w Roster
sel_feat = avg_corrs[avg_corrs["Avg corr (Abs)"] > .3].index.values.tolist()
sel_feat
```




    ['draft_Games',
     'draft_Minutes.Played',
     'draft_PTS',
     'draft_Pk',
     'draft_attend_college']



## Cross-Validation


```python
X = data[sel_feat]
y = data["Roster"]
```


```python
predicted = cross_val_predict(LogisticRegression(), X, y, cv = 10)
acc_score = metrics.accuracy_score(y, predicted)
roster_vars = ["Did not play", "Played"]

print(f"Accuracy score: {round(acc_score, 2)}\n")
print("Classification report")
print(metrics.classification_report(y, predicted, target_names = roster_vars))
```

    Accuracy score: 0.86
    
    Classification report
                  precision    recall  f1-score   support
    
    Did not play       0.76      0.69      0.72       234
          Played       0.89      0.92      0.91       655
    
     avg / total       0.86      0.86      0.86       889
    
    


```python
# Get coefficients
clf = LogisticRegression()
clf.fit(X, y)
clf.coef_
```




    array([[ 1.81319323e-02, -7.66725676e-04,  6.39905653e-04,
            -4.72543116e-02,  1.50353549e+00]])




```python
# Predicted probabilities of all players

pp = pd.DataFrame(clf.predict_proba(X))

pl = orig_data[["draft_Player", "draft_Draft_Yr"]]

pl_sel_feat = orig_data[sel_feat]

pred = pd.DataFrame(predicted)
pred.columns = ["Roster (Pred)"]

pl_rost = orig_data["Roster"]

pl_pp = pd.concat([pl, pl_sel_feat, pl_rost, pred, pp], axis = 1)

pl_pp.rename(columns = 
             {0: "Did not play (PP)",
              1: "Played (PP)",
              "Roster": "Roster (Actual)"}, inplace = True)

pl_pp.to_csv("predictions/draft00_15_preds.csv")

pl_pp.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>draft_Player</th>
      <th>draft_Draft_Yr</th>
      <th>draft_Games</th>
      <th>draft_Minutes.Played</th>
      <th>draft_PTS</th>
      <th>draft_Pk</th>
      <th>draft_attend_college</th>
      <th>Roster (Actual)</th>
      <th>Roster (Pred)</th>
      <th>Did not play (PP)</th>
      <th>Played (PP)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Speedy Claxton</td>
      <td>2000</td>
      <td>334</td>
      <td>8548</td>
      <td>3096</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.075174</td>
      <td>0.924826</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mark Karcher</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.573603</td>
      <td>0.426397</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Stromile Swift</td>
      <td>2000</td>
      <td>547</td>
      <td>10804</td>
      <td>4582</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.001588</td>
      <td>0.998412</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jamaal Magloire</td>
      <td>2000</td>
      <td>680</td>
      <td>14621</td>
      <td>4917</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.004775</td>
      <td>0.995225</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Erick Barkley</td>
      <td>2000</td>
      <td>27</td>
      <td>266</td>
      <td>77</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.272217</td>
      <td>0.727783</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save classifier
import pickle
pickle.dump(clf, open("final_classifier.pkl", "wb"))
```

## Testing Classifier on New Data


```python
new_data = pd.read_csv("../data/clean/draft16_nba17.csv")
var_list = sel_feat
var_list.extend(("draft_Player", "Roster"))
new_data = new_data[var_list].dropna(axis = 0, how = "any")
```


```python
X_new = new_data.drop(["Roster", "draft_Player"], axis = 1)
y_new = new_data["Roster"]
```


```python
classifier = pickle.load(open("final_classifier.pkl", "rb"))
```


```python
# Prediction
new_pred = classifier.predict(X_new)
pred_act = pd.DataFrame({"Roster (Pred)": new_pred,
                         "Roster (Actual)": y_new})
```


```python
# New predicted probabilities
new_pp = pd.DataFrame(classifier.predict_proba(X_new))
new_pl_sel_feat = new_data[var_list].drop(["Roster"], axis = 1)
new_pl_pp = pd.concat([new_pl_sel_feat, pred_act, new_pp], axis = 1)
new_pl_pp.rename(columns = 
                  {0: "Did not play (PP)",
                   1: "Played (PP)"}, inplace = True)

col_order = ['draft_Player',
 'draft_Games',
 'draft_Minutes.Played',
 'draft_PTS',
 'draft_Pk',
 'draft_attend_college',
 'Roster (Actual)',
 'Roster (Pred)',
 'Did not play (PP)',
 'Played (PP)']
new_pl_pp = new_pl_pp[col_order]

new_pl_pp.to_csv("predictions/draft16_preds.csv")

new_pl_pp.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>draft_Player</th>
      <th>draft_Games</th>
      <th>draft_Minutes.Played</th>
      <th>draft_PTS</th>
      <th>draft_Pk</th>
      <th>draft_attend_college</th>
      <th>Roster (Actual)</th>
      <th>Roster (Pred)</th>
      <th>Did not play (PP)</th>
      <th>Played (PP)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ben Simmons</td>
      <td>81.0</td>
      <td>2732.0</td>
      <td>1279.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.107473</td>
      <td>0.892527</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brandon Ingram</td>
      <td>138.0</td>
      <td>4254.0</td>
      <td>1689.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.099886</td>
      <td>0.900114</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jaylen Brown</td>
      <td>148.0</td>
      <td>3493.0</td>
      <td>1532.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.056489</td>
      <td>0.943511</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dragan Bender</td>
      <td>125.0</td>
      <td>2643.0</td>
      <td>677.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.278415</td>
      <td>0.721585</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kris Dunn</td>
      <td>130.0</td>
      <td>2858.0</td>
      <td>992.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.073372</td>
      <td>0.926628</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_acc_score = metrics.accuracy_score(y_new, new_pred)
roster_vars = ["Did not play", "Played"]

print(f"Accuracy score: {round(new_acc_score, 2)}\n")
print("Classification report")
print(metrics.classification_report(y_new, new_pred, target_names = roster_vars))
```

    Accuracy score: 0.69
    
    Classification report
                  precision    recall  f1-score   support
    
    Did not play       0.17      0.22      0.19         9
          Played       0.83      0.78      0.80        45
    
     avg / total       0.72      0.69      0.70        54
    
    
