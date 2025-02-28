# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

<!-- Model Card -->

## Model Details

**Type:** Logistic Regression classifier  
**Purpose:** Binary classification to predict whether an individual earns >50K or <=50K  

**Input Features:**
- **Continuous:** age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week  
- **Categorical:** workclass, education, marital-status, occupation, relationship, race, sex, native-country  

**Preprocessing:**
- Categorical features are one-hot encoded  
- The target variable (salary) is binarized using a label binarizer  

**Hyperparameters:**
- Default Logistic Regression parameters were used with an increased `max_iter` (attempted 1000 to 2000) to address convergence issues; however, a `ConvergenceWarning` was still observed.  

**Baseline:** This model serves as a baseline for further experimentation and tuning.  

---

## Intended Use

**Primary Use Case:**  
- Educational purposes as part of a machine learning pipeline project in a Udacity nanodegree program.  
- The model predicts income level (above or below 50K) based on demographic and employment data.  

**Limitations:**  
- Not intended for production-level decision making without further validation, tuning, and fairness assessments.  

---

## Training Data

**Source:** Publicly available Census Bureau data (`census.csv`)  
**Size:** 32,561 records with 15 features  

**Preprocessing:**  
- Data was split into training and test sets (e.g., an 80/20 split).  
- The training set was processed (categorical encoding and label binarization) using the provided `process_data` function.  

---

## Evaluation Data

**Method:**  
- The test set (approximately 20% of the data) was processed in inference mode using the fitted encoder and label binarizer.  

**Overall Performance Metrics (on test data):**  
- **Precision:** ~0.73  
- **Recall:** ~0.60  
- **F1 Score:** ~0.66  

---

## Metrics

**Metrics Used:** Precision, Recall, and F1 Score (`beta=1`).  

**Overall Performance:**  
- **Precision:** `0.7292`  
- **Recall:** `0.6015`  
- **F1 Score:** `0.6592`  

**Performance on Data Slices:**  
Slice metrics were computed for each categorical feature. Here are some examples:

- **Workclass:**
  - Federal-gov (191 samples): **Precision ~0.74**, **Recall ~0.61**, **F1 ~0.67**  
  - Private (4,578 samples): **Precision ~0.78**, **Recall ~0.51**, **F1 ~0.62**  
  - Self-emp-not-inc (498 samples): **Precision ~0.58**, **Recall ~0.55**, **F1 ~0.57**  

- **Education:**
  - Bachelors (1,053 samples): **Precision ~0.71**, **Recall ~0.80**, **F1 ~0.75**  
  - HS-grad (2,085 samples): **Precision ~0.77**, **Recall ~0.17**, **F1 ~0.28** (indicating low recall for this subgroup)  

- **Native-country:**
  - United-States (5,870 samples): **Precision ~0.74**, **Recall ~0.54**, **F1 ~0.63**  
  
_(Additional slice metrics are provided in the attached `slice_output.txt`.)_  

These results illustrate that while overall performance is reasonable, the model’s effectiveness can vary significantly across subgroups.  

---

## Ethical Considerations

**Data Sensitivity:**  
- The Census data contains sensitive demographic information. Care must be taken to avoid reinforcing potential biases present in the data.  

**Fairness:**  
- The variability in performance across slices (e.g., some subgroups achieving very low recall) highlights potential fairness concerns.  

**Usage Caution:**  
- This model is developed for educational purposes and should not be used for real-world decisions without comprehensive bias, fairness, and performance analysis.  

---

## Caveats and Recommendations

**Convergence:**  
- A `ConvergenceWarning` was observed during training, suggesting that further preprocessing (such as feature scaling) or hyperparameter tuning might be needed.  

**Performance Variability:**  
- The model shows variable performance across different demographic slices. Further investigation and potentially additional techniques (e.g., resampling or specialized algorithms) may be necessary to improve performance for underrepresented groups.  

**Future Improvements:**  
- Consider exploring alternative algorithms and hyperparameter tuning to improve overall metrics and reduce subgroup performance discrepancies.  

 

