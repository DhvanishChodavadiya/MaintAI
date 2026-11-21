MaintAI — Project Progress Documentation (Week 1–2)
Project: Agentic Industrial Maintenance & Operations Copilot
Status: Environment setup, dataset acquisition and initial exploratory data analysis completed.
1. Project objective
MaintAI is planned as an evidence-grounded decision-support assistant for industrial maintenance engineers. It will combine technical-document retrieval, structured maintenance records, sensor analytics and (where appropriate) machine-learning predictions. The system is intended to present evidence, citations and uncertainty—not to autonomously control machinery or replace engineering judgement.
2. Current status
- Conda environment: MaintAI
- Python: 3.13.18
- Dataset: AI4I 2020 Predictive Maintenance
- Dataset shape: 10,000 rows × 14 columns
- Missing values: 0
- Duplicate rows: 0
- Duplicate UDI values: 0
- Duplicate Product IDs: 0
- Initial EDA charts generated: failure distribution, failure types, sensor distributions and correlation heatmap.
3. Dataset fields
Field	Role
UDI	Unique observation identifier
Product ID	Product identifier
Type	Product quality variant (L, M, H)
Air temperature [K]	Sensor measurement
Process temperature [K]	Sensor measurement
Rotational speed [rpm]	Sensor measurement
Torque [Nm]	Sensor measurement
Tool wear [min]	Sensor measurement
Machine failure	Primary binary target for initial modelling
TWF, HDF, PWF, OSF, RNF	Individual failure-mode labels


Leakage warning: Do not use the failure-mode labels as input features when predicting Machine failure. They directly encode information about the target.
4. Data quality and distributions
Check	Result
Rows	10,000
Columns	14
Missing values	0
Duplicate rows	0
Duplicate UDI	0
Duplicate Product IDs	0


Product type distribution
Type	Count	Percentage
L	6,000	60.00%
M	2,997	29.97%
H	1,003	10.03%


Overall machine-failure distribution
Label	Count	Percentage
No failure (0)	9,661	96.61%
Failure (1)	339	3.39%


The target is strongly imbalanced. Accuracy alone is not sufficient for evaluating a failure classifier. Later evaluation should include failure-class precision, recall, F1-score and a confusion matrix; PR-AUC may also be useful.
Individual failure labels
Label	Count	Percentage of all rows
TWF	46	0.46%
HDF	115	1.15%
PWF	95	0.95%
OSF	98	0.98%
RNF	19	0.19%


These counts sum to 373, exceeding the 339 overall failure observations. The labels must not be assumed mutually exclusive; multiple failure-mode labels may occur on one observation. Verify this with a label-consistency check.
5. Correlation observations
Pair / feature vs target	Pearson correlation
Air temperature vs Process temperature	0.88
Rotational speed vs Torque	-0.88
Torque vs Machine failure	0.19
Tool wear vs Machine failure	0.11
Air temperature vs Machine failure	0.08
Process temperature vs Machine failure	0.04
Rotational speed vs Machine failure	-0.04


Correlation is descriptive, does not establish causation and captures linear association only. A low pairwise correlation does not prove a feature is useless.
6. Visual artefacts
The EDA script generated these figures:
- docs/figures/failure_distribution.png
- docs/figures/failure_types.png
- docs/figures/sensor_distributions.png
- docs/figures/correlation_heatmap.png
7. Limitations
- AI4I 2020 is synthetic; results do not establish real-factory performance.
- The current dataset does not provide a verified chronological history for a particular machine. Do not describe rows as a multi-day machine history without verified time and machine identifiers.
- No repeated Product IDs were observed in the inspected file; do not assume Product ID is a persistent machine identifier.
- Class-aware evaluation and leakage-safe splitting are required.
8. Completed work
1. Created MaintAI project structure and initial GitHub commit.
2. Configured Conda environment with Python 3.13.18.
3. Downloaded and loaded AI4I 2020 CSV.
4. Inspected dimensions, columns, types, missing values and descriptive statistics.
5. Checked duplicate rows and identifiers.
6. Calculated product-type and failure distributions.
7. Generated four EDA visualisations.
9. Next steps
1. Verify failure-label consistency and inspect potential outliers.
2. Build a reusable data-validation and preprocessing pipeline.
3. Define a leakage-safe train/test strategy.
4. Establish a baseline model and evaluate failure-class precision, recall and F1.
5. Later integrate predictive outputs into the LangGraph-based MaintAI application.
10. Current scripts
- notebooks/download_dataset.py
- notebooks/01_dataset_exploration.py
- notebooks/02_eda_analysis.py
Documentation note: This file records completed work and observed results from the current project stage. Planned application features are not represented as implemented.