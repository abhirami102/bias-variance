# Bias–Variance Investigation — Team 14

| Name | Roll No. |
|------|----------|
| Abhirami Balakrishnan | SJC24CS005 |
| Alan Xavier Selbi | SJC24CS030 |
| Joseph Francis | SJC24CS137 |
| Sidharth S | SJC24CS200 |

**College:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)
**Department:** Computer Science and Engineering
**Course:** 24SJPCCST503 – Machine Learning
**Academic Year:** 2026–2027

---

## Investigation Topic

**High Bias and High Variance (Bias–Variance Trade-off)**

## Research Question

> How can experimental evidence reveal high bias and high variance?

## Operational Question

> How do the training and testing performance of a Decision Tree Regressor change as model complexity (`max_depth`) increases, and what do those changes reveal about underfitting (high bias) and overfitting (high variance) on the UCI Student Performance dataset?

---

## 1. Overview

This project is an experimental ML investigation, not a production grade-prediction system. We train four Decision Tree Regressors that differ **only** in `max_depth`, while keeping everything else fixed. The goal is not to build the most accurate grade predictor, but to use student data as a setting for a controlled study of how model complexity affects generalization.

Bias and variance are **not** computed as separate numerical quantities. They are **inferred** from observable behaviour:

- Poor training performance → model too limited to capture the pattern (**high bias**)
- Very strong training performance + much poorer testing performance → model learned sample-specific details (**high variance**)

**Reasoning chain:** Question → Hypothesis → Experiment → Evidence → Analysis → Conclusion

---

## 2. Hypothesis

| ID | Hypothesis |
|----|------------|
| H1 | A very simple Decision Tree will have limited ability to fit the data, giving evidence of underfitting and relatively high bias. |
| H2 | As the tree becomes more complex, training performance will improve. |
| H3 | At excessive complexity, training performance may become extremely high while testing performance stops improving or becomes worse, giving evidence of high variance and overfitting. |

> These are hypotheses to be tested — not conclusions. The final answer is based on experimental evidence.

---

## 3. Dataset

**UCI Student Performance Dataset** (Cortez, 2008)

- Secondary-school student achievement in two Portuguese schools
- File used: `student-mat.csv` (Mathematics) — **395 observations**, stored in the `data/` folder
- Portuguese-language file **not** used, because the target is the final Mathematics grade
- **Target:** `G3` — final grade, scale 0–20
- `G3` was excluded from the input features to avoid **target leakage**

**Input features (10):**

| Feature | Meaning |
|---------|---------|
| `studytime` | Weekly study time (1: <2 h; 2: 2–5 h; 3: 5–10 h; 4: >10 h) |
| `absences` | Number of school absences |
| `failures` | Number of previous class failures |
| `G1` | First-period grade (0–20) |
| `G2` | Second-period grade (0–20) |
| `age` | Student age |
| `Medu` | Mother's education level (0: none to 4: higher education) |
| `Fedu` | Father's education level (0: none to 4: higher education) |
| `goout` | How often the student goes out with friends (1: very low to 5: very high) |
| `health` | Current health status (1: very bad to 5: very good) |

These are treated as predictive inputs, **not** as proven causes of the final grade.

---

## 4. Experimental Design

**Independent variable:** Decision Tree complexity, controlled through `max_depth`.

| Experiment | `max_depth` | Purpose |
|------------|-------------|---------|
| 1 | 1 | Very shallow tree — expected underfitting |
| 2 | 3 | Intermediate tree — expected best balance |
| 3 | 7 | Deep tree — expected onset of overfitting |
| 4 | 15 | Very deep tree — expected strong overfitting |

**Controlled variables:** same dataset, same ten features, same target (`G3`), same algorithm, same evaluation metrics, same train–test split.

| Component | Setting |
|-----------|---------|
| Dataset | UCI Student Performance (Mathematics), `student-mat.csv`, 395 observations |
| Target | `G3`, final Mathematics grade (0–20) |
| Model | `DecisionTreeRegressor(max_depth=depth, random_state=42)` (scikit-learn) |
| Varied variable | `max_depth` = 1, 3, 7, 15 |
| Split | 80% training (316) / 20% testing (79), `random_state=42` |
| Preprocessing | No feature scaling — tree splits are threshold comparisons, so rescaling does not change them |

**Why four depths?** At least three conditions were required; four were enough to show the progression from low to high complexity. An initial AI-suggested setup of eight depths was reduced to four.

---

## 5. Model

| Model | Settings | Why |
|-------|----------|-----|
| Decision Tree Regressor | `max_depth` ∈ {1, 3, 7, 15}, `random_state=42` | `max_depth` gives direct control over complexity: a shallow tree makes a few coarse splits, a deep tree makes finer splits and can fit individual training observations closely. |

---

## 6. Metrics

| Metric | Interpretation |
|--------|----------------|
| Training MSE | Average squared error on the training set. Lower = better fit to training data. |
| Testing MSE | **Primary metric** — average squared error on unseen data. Lower = better. |
| Training R² | Variance explained on training data (1.0 = perfect fit on that data only). |
| Testing R² | Variance explained on unseen data. Higher = better. |
| MSE Gap | Testing MSE − Training MSE. A larger positive gap indicates more overfitting. |

---

## 7. Key Results

| Max Depth | Training MSE | Testing MSE | Training R² | Testing R² | MSE Gap |
|-----------|-------------:|------------:|------------:|-----------:|--------:|
| 1  | 10.158 | 8.096 | 0.516 | 0.605 | −2.061 |
| 3  | 2.547  | **4.703** | 0.879 | **0.771** | 2.156 |
| 7  | 0.169  | 6.052 | 0.992 | 0.705 | 5.883 |
| 15 | 0.000  | 6.177 | 1.000 | 0.699 | 6.177 |

**Step-by-step change summary:**

| Comparison | Δ Training MSE | Δ Testing MSE | Effect |
|------------|---------------:|--------------:|--------|
| Depth 1 → 3  | −7.611 | −3.393 | Both improve — bias reduced — H1, H2 ✅ |
| Depth 3 → 7  | −2.378 | +1.349 | Training improves, testing worsens — overfitting begins — H3 ✅ |
| Depth 7 → 15 | −0.169 | +0.125 | Training perfect, testing flat/worse — H3 ✅ |

The notebook produces three figures: Training vs Testing MSE by depth, Training vs Testing R² by depth, and the Training–Testing MSE gap by depth.

---

## 8. Main Findings

1. **Depth 1 — limited fit, relatively high bias (H1 supported).** A single split gives only Training R² = 0.516 and Training MSE = 10.158, far weaker than every deeper model. This is consistent with underfitting.

2. **Depth 3 — best generalization (H2 supported).** Lowest Testing MSE (4.703) and highest Testing R² (0.771), with a moderate gap of 2.156. Going from depth 1 to 3 improved both training and testing performance, so reducing bias was beneficial at this stage.

3. **Depths 7 and 15 — high variance and overfitting (H3 supported).** Training performance kept improving (Training R² 0.992 → 1.000) while testing performance did not (Testing R² 0.705 → 0.699). The depth-15 tree fitted the training set perfectly, yet Testing MSE was 6.177. The MSE gap widened to 5.883 and 6.177, the strongest evidence of overfitting.

4. **The negative gap at depth 1 is not evidence of low bias.** Testing MSE (8.096) was lower than Training MSE (10.158) because this particular test subset happened to be easier to predict. The stronger evidence for high bias is the poor training fit.

5. **Bias–variance trade-off observed.** Increasing complexity reduced bias at first, but excessive complexity increased the train–test gap. This does **not** mean deeper trees are always worse — only that, for this dataset and split, very high depth did not improve testing performance.

---

## 9. Limitations

- **Single dataset** — only the Mathematics data from two Portuguese schools (395 observations); findings may not generalize.
- **Single train–test split** (`random_state=42`) — a different split could give different values; cross-validation would give a stronger estimate of variability.
- **Strong predictors G1 and G2** — strongly related to G3, which makes prediction easier; the study is not about identifying causal factors.
- **Single algorithm** — only a Decision Tree; other model families may behave differently.
- **Limited depth values** — only four depths tested; depth 3 was best only among those four, and the exact onset of overfitting was not identified.
- **Bias and variance not directly computed** — inferred from training/testing behaviour only.
- **Small difference between deep models** — depth 7 (6.052) and depth 15 (6.177) should not be over-interpreted.
- **Predictive association only** — no claim of causation between any feature and the final grade.

---

## 10. Repository Structure

```
bias-variance/
├── README.md                              ← This file
├── bias-variance-investigation.ipynb      ← PRIMARY NOTEBOOK (code, plots, analysis)
└── data/
    └── student-mat.csv                    ← UCI Student Performance (Mathematics)
```

The notebook is the primary implementation artifact. It trains one `DecisionTreeRegressor` per depth, computes the metrics above on the training and testing sets, and plots the results.

---

## 11. How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Get the code and data

```bash
git clone https://github.com/abhirami102/bias-variance.git
cd bias-variance
```

The dataset is in the `data/` folder. If it is missing, download it from the UCI Machine Learning Repository ([Student Performance](https://doi.org/10.24432/C5TG7T)) and use the Mathematics file `student-mat.csv`. Note that the UCI file is semicolon-separated (`sep=";"`).

### Run the Notebook

Launch Jupyter from the project root:

```bash
jupyter notebook bias-variance-investigation.ipynb
```

Then: **Kernel → Restart & Run All**

If the notebook cannot find the data, update the path in the cell that loads `student-mat.csv`.

---

## 12. AI Usage

ChatGPT was used as a supporting tool for understanding machine-learning concepts, planning the controlled experiment, generating and explaining code, troubleshooting issues, and organizing the documentation. The experiment was executed by the team in Jupyter Notebook on the selected dataset, and all numerical results and graphs come from the team's own runs and were checked before the final interpretation was written. One AI suggestion was modified: an initial setup of eight depth values was reduced to four meaningful levels (1, 3, 7 and 15). The team remains responsible for the methodology and conclusions.

---

## 13. References

1. Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
3. Scikit-learn documentation. `DecisionTreeRegressor` and `train_test_split`.
4. Scikit-learn documentation. `mean_squared_error` and `r2_score`.
5. Course Project Guidelines, 24SJPCCST503 Machine Learning, Academic Year 2026–2027, Team 14: Bias–Variance investigation.
