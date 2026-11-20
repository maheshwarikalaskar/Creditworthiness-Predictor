# Credit Scoring Predictor

A Streamlit app that estimates whether a loan applicant is a good or bad credit risk, backed by a scikit-learn pipeline trained on the German Credit dataset. The training notebook compares several models, chooses the decision threshold using the dataset's own cost matrix, explains the model, and audits it for group differences.

> **Disclaimer:** This is an educational project on a small, old dataset. It is not a lending tool and must not be used for real credit decisions.

## Dataset

The German Credit dataset (OpenML `credit-g`, from the UCI Statlog collection): 1,000 loans from 1994 described by 20 attributes (amounts are in Deutsche Marks), with a `good` or `bad` label. 70% of the loans are good, so a model that approves everyone is already 70% accurate. The notebook reports that baseline next to every model.

## Approach

1. **Features:** 7 numeric and 11 categorical inputs, plus an engineered `credit_per_month` feature (credit amount divided by duration + 1), which lives inside the pipeline so the notebook and the app can never disagree. Sensitive attributes `personal_status` (sex and marital status) and `foreign_worker` are excluded from the model.
2. **Models:** Logistic Regression, Random Forest and Gradient Boosting (with and without class weights) against a majority-class baseline, compared with repeated stratified cross-validation on the training split.
3. **Decision threshold:** the dataset's documentation says approving a bad borrower costs 5 times as much as rejecting a good one. The notebook picks the threshold that minimises that cost on out-of-fold predictions, then evaluates it once on a held-out test set.
4. **Evaluation:** ROC-AUC, accuracy, recall on bad loans and cost per applicant, compared with approving everyone and rejecting everyone.
5. **Explainability:** permutation importance on the held-out set.
6. **Fairness audit:** approval rate, actual good rate and share of bad loans caught for groups defined by age, foreign-worker status and personal status, including the excluded attributes.

The notebook prints every number above when you run it. The headline test metrics are also saved in `model_info.json`.

## Note on an earlier version

The first version of the notebook selected categorical columns with `select_dtypes(include='object')`. OpenML returns them with `category` dtype, so the list was empty and the model trained on the 8 numeric columns only. The current notebook selects columns by exclusion and asserts that every input column reaches the model.

## Run the app

```bash
git clone https://github.com/maheshwarikalaskar/Creditworthiness-Predictor.git
cd Creditworthiness-Predictor
pip install -r requirements.txt
streamlit run app.py
```

The app needs `credit_model.pkl` and `model_info.json`, which the notebook creates. Dropdown choices and number limits come from the training data, so the app only offers values the model has seen.

## Retrain

Open `credit_score_model_training.ipynb` in Colab or Jupyter and run all cells. It downloads the dataset once (and saves it as `credit_g.csv`), trains and evaluates the models, and writes `credit_model.pkl`, `model_info.json`, `credit_features.py` and a `requirements.txt` pinned to the scikit-learn version that saved the model.

## Files

```
.
├── app.py                              # Streamlit app
├── credit_features.py                  # shared feature engineering
├── credit_score_model_training.ipynb   # training, evaluation, fairness audit
├── credit_model.pkl                    # created by the notebook
├── model_info.json                     # input schema, threshold, metrics (created by the notebook)
├── credit_g.csv                        # dataset snapshot (created by the notebook)
├── requirements.txt
└── README.md
```

## Limitations

- 1,000 loans from 1994 in Germany: not representative of lending today or in other countries.
- The model score is not a calibrated probability.
- Excluding sensitive attributes does not remove bias: other features can act as proxies, which is why the notebook audits the groups directly. The groups are small, so the audit shows direction, not proof.
- A statistical demo with no regulatory review of any kind.

## Future work

- Probability calibration and SHAP-based explanations in the app
- A larger, more recent credit dataset
- Formal fairness metrics and mitigation

## Author

**Maheshwari Kalaskar**: [GitHub](https://github.com/maheshwarikalaskar) · [LinkedIn](https://www.linkedin.com/in/maheshwarikalaskar/)
