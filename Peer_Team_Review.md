## Peer Programming Review of KNN Classification on Palmer Penguins



### Overall assessment
The peer team developed a well-structured K-nearest neighbours classifier for predicting penguin species. Its strongest implementation decision was combining preprocessing and classification in one pipeline and fitting that pipeline only on the training data.
The saved output reports 98.55% test accuracy, equivalent to 68 correct predictions out of 69. This supports KNN as a promising model for this dataset. 

However, a single train–test split, an unvalidated choice of five neighbours, and the absence of comparisons with other algorithms mean that KNN has not been demonstrated to be the best model.

Our review also identifies a reproducibility issue: the notebook calls its exploratory analysis function before the dataset is loaded in the displayed execution sequence.

## Information shared by the peer team

**Dataset choice and prediction task**
The peer team selected the Palmer Penguins dataset, loaded through seaborn.load_dataset('penguins'). The objective is multiclass classification: predicting whether a penguin is Adelie, Chinstrap, or Gentoo.
They record 344 observations and seven columns: six predictors and one target.

| Component | Details |
|---|---|
| Target | species |
| Numerical predictors | Bill length, bill depth, flipper length, and body mass |
| Categorical predictors | Island and sex |
| Species counts | Adelie: 152; Gentoo: 124; Chinstrap: 68 |
| Missing numerical values | Two missing entries in each numerical column |
| Missing categorical values | Eleven missing entries in sex; none in island |
| Missing target values | None reported |
| Duplicate rows | None reported |


**Our team’s insight**: This is a useful educational dataset because it requires meaningful preprocessing while remaining small enough to inspect individual predictions. Its numerical measurements provide a plausible basis for similarity comparisons. Nevertheless, similarity depends on how the features are scaled and weighted; dataset suitability should be tested rather than assumed.


**Preprocessing decisions**

| Decision | Implementation | Our assessment |
|---|---|---|
| Numerical imputation | Replace missing values with training-set medians | Reasonable baseline that retains observations, but can create artificial similarities |
| Categorical imputation | Replace missing categories with the training-set mode | Simple and practical, but hides uncertainty about missing sex values |
| Numerical scaling | Apply StandardScaler after imputation | Important because raw body-mass differences would otherwise dominate distances |
| Categorical encoding | Apply one-hot encoding to island and sex | Avoids assigning an artificial numerical order to categories |
| Unseen categories | Use handle_unknown='ignore' | Prevents prediction errors, but does not establish reliability for unfamiliar categories |
| Data splitting | Stratified 80% training and 20% testing; seed 42 | Preserves approximate class proportions and makes the split repeatable |
| Leakage prevention | Fit the complete pipeline only on training data | Correctly restricts learned preprocessing statistics to training observations |

The resulting split contains 275 training observations and 69 test observations.

**Our team’s insight**: Preprocessing is part of the KNN model because it determines which observations become neighbours. Standardisation improves comparability across numerical units, but does not guarantee that every feature deserves equal influence.

**Model configuration and saved results**

The classifier explicitly sets n_neighbors=5. Other classifier settings are left at their defaults rather than justified through experiments.
The following values reproduce the rounded classification report 

| Species | Precision | Recall | F1 score | Test support |
|---|---:|---:|---:|---:|
| Adelie | 1.00 | 0.97 | 0.98 | 30 |
| Chinstrap | 0.93 | 1.00 | 0.97 | 14 |
| Gentoo | 1.00 | 1.00 | 1.00 | 25 |
| Macro average | 0.98 | 0.99 | 0.98 | 69 |
| Weighted average | 0.99 | 0.99 | 0.99 | 69 |

Overall accuracy is reported separately as 0.9855.
The class metrics and supports imply this confusion matrix:

| Actual species | Predicted Adelie | Predicted Chinstrap | Predicted Gentoo |
|---|---:|---:|---:|
| Adelie | 29 | 1 | 0 |
| Chinstrap | 0 | 14 | 0 |
| Gentoo | 0 | 0 | 25 |

An always-Adelie classifier would correctly classify 30 of the 69 test observations, giving approximately 43.48% accuracy. The reported KNN result is substantially above that simple baseline.

**Our team’s insight:** The direction of the error matters. One actual Adelie was predicted as Chinstrap, reducing Adelie recall and Chinstrap precision. Perfect performance on the 25 Gentoo test observations does not establish perfect performance on future Gentoo observations.

#### Some interview questions and answers

**Question 1 — Why did you choose this dataset?**

**Peer-team answer:** We chose Palmer Penguins because it provides a realistic classification task with numerical measurements, categorical predictors, and missing values. It lets us demonstrate a complete preprocessing and classification pipeline.

**Our team’s insight:** The choice supports the learning objective. 

**Question 2 — Why did you impute missing values instead of deleting rows?**

**Peer-team answer:** We used median and most-frequent imputation to retain observations in a relatively small dataset.

**Our team’s insight:** Retaining observations is useful, but imputation changes the neighbourhood structure. A penguin with several missing measurements may become artificially close to typical observations. We would compare this approach with complete-case analysis and consider missingness indicators, using training-data validation.

**Question 3 — Why is scaling important for KNN?**

**Peer-team answer:** KNN uses distances. Body mass is measured in thousands of grams, whereas bill measurements are in tens of millimetres. Standardisation prevents numerical units from determining most of the distance.

**Our team’s insight:** The reasoning is correct. However, the notebook does not show an unscaled comparison, so the size of scaling’s performance benefit has not been measured. Saying other features become completely “useless” without scaling would be too strong.

**Question 4 — How did you prevent data leakage?**

**Peer-team answer:** We split the data before fitting the pipeline. Imputation, scaling, and encoding were fitted using the training set, and the fitted transformations were then applied to the test set.

**Our team’s insight: **This is a major strength. During future cross-validation, the entire pipeline must also be fitted separately within each training fold. Preprocessing the complete dataset before cross-validation would reintroduce leakage.

**Question 5 — Why did you choose five neighbours?**

**Peer-team answer:** The notebook uses five neighbours as its selected configuration. It does not include a tuning experiment showing that five is optimal.

**Our team’s insight: **Small values of k can be sensitive to noise, while larger values can smooth away useful local differences. We recommend comparing several values through stratified cross-validation on the training set. An odd k does not eliminate all possible voting ties in a three-class problem.
**
Question 6— Was KNN the best choice?**

**Peer-team answer:** KNN achieved a strong saved test result, but the notebook does not compare it with other algorithms.

**Our team’s insight:** KNN is a defensible baseline for this small dataset. 
	


**Was KNN the best choice for this dataset**

Our conclusion is that KNN was appropriate and promising, but not established as the best choice.

A stronger evaluation would retain a final holdout and compare complete pipelines using stratified cross-validation on the training data. Candidate models could include a majority-class baseline, logistic regression, a decision tree, and a random forest.

The existing test result must not become a target that the team repeatedly tunes against. Further claims based on repeated reuse of that test set would require appropriate qualification or fresh evaluation data.

**Possible MLOps extensions**

The code presents MLOps ideas, but does not implement an operational system. A proportionate extension would include the following.

| Area | Proposed extension | Purpose |
|---|---|---|
| Reproducibility | Version the dataset, code, dependencies, and split configuration | Make results repeatable |
| Data validation | Check schema, units, missingness, categories, and plausible ranges | Detect unsuitable inputs before prediction |
| Experiment tracking | Record parameters, validation results, and model artifacts | Support auditable comparisons |
| Automated checks | Test fresh notebook execution and pipeline predictions | Catch broken dependencies and workflow errors |
| Deployment | Save and serve the complete fitted pipeline | Keep preprocessing consistent with training |
| Monitoring | Track missing values, unfamiliar categories, feature distributions, and latency | Detect operational changes |
| Performance monitoring | Evaluate accuracy and class recall when verified labels arrive | Establish whether predictive quality has changed |
| Model updates | Validate candidate replacements and retain rollback capability | Prevent untested updates from replacing a working model |
Our team would avoid assuming that daily data collection or retraining is necessary for this small dataset. Update frequency should reflect actual data availability and use.


Final team reflection
The peer team’s main achievement is a sensible preprocessing pipeline that supports strong performance on its selected test set. Our review reinforces that preprocessing decisions directly shape KNN behaviour and deserve as much attention as the classifier.
The main improvements are to establish fresh-session reproducibility, validate model and feature choices, investigate the classification error, and replace peer-comparison placeholders with verified evidence. These steps would turn a successful demonstration into a more convincing and reproducible machine-learning study.

