# 🏠🔁 Transfer Learning for Tehran House Price Prediction

Knowledge transfer between two Tehran housing price datasets using a shared MLP architecture: using a larger dataset (Source) to improve prediction on a smaller dataset (Target), with staged layer freezing and fine-tuning.

## 📋 Project Overview

The central question of the project is: **can the knowledge a neural network learns from a large dataset (139,749 records) help improve prediction on a smaller, related dataset (3,473 records)?**

To answer this question:
1. A model is trained on the large dataset (**Source**) using only the 4 features shared between the two datasets.
2. An independent baseline model with the same architecture is trained only on the small dataset (**Target**), for comparison.
3. The early layers of the Source model (which learned more general patterns) are transferred into a new model, while the final layers come from the Target baseline model.
4. This combined model is fine-tuned on the Target data in two stages: first with the transferred layers frozen, then with all layers fully unfrozen.
5. Finally, the performance of the final model (Transfer) is compared against the independent baseline to determine whether knowledge transfer was actually beneficial.

## 📁 Project Structure

| File | Role |
| --- | --- |
| `source.py` | Full pipeline on the Source dataset (`en_data.csv`): preprocessing, feature engineering, training the full model, and finally exporting the 4 shared features (`transfer_x`) for use in Transfer Learning |
| `target.py` | Full pipeline on the Target dataset (`tehranHprice.csv`): similar to the above, also exporting the same 4 shared features |
| `main.py` | Main Transfer Learning script: imports the two files above, builds the Source/Target baseline models, transfers the layers, performs two-stage fine-tuning, and compares the final results |

> 📝 Important note about execution: because `main.py` loads these two files with `import source` and `import target`, and Python executes all top-level code of a module at import time, **running `main.py` automatically also runs the complete independent pipelines of both `source.py` and `target.py`**. That means both the independent full models are trained and the Transfer Learning process is carried out. This is why you will see four separate report blocks in the final output (the full Source model, the full Target model, and then the Transfer sections).

## 🧩 Shared Feature Engineering

Despite differences in their columns, both datasets use a shared logic for encoding geographic location (exactly the same approach we saw in the previous Tehran house price project):

- **`district_number`**: A manual mapping of neighborhoods to municipal district numbers (1 to 22), because this information does not exist in the raw datasets, yet it is one of the strongest price drivers in the real market. The Source dataset even includes the Persian names of neighborhoods, and its mapping is broader than the Target's.
- **`is_suburb`**: A binary flag for suburban neighborhoods / those outside Tehran's official district system.
- **`top_area`**: A feature for identifying the premium segment of the market, but **its condition intentionally differs between the two datasets**:
  - In `source.py`: only District 1 with an area greater than 50 m². According to the code's own comment, the initial condition also included Districts 2 and 3, but those two districts introduced significant noise into the model and reduced accuracy, so they were removed.
  - In `target.py`: Districts 1, 2, or 3 with an area greater than 120 m².

  This difference shows that the feature was tuned through independent experimentation on each dataset, not copied as a general rule.
- **Target Encoding without data leakage**: Both scripts compute the mean price of each district with smoothing (to reduce the effect of low-data districts), and build these statistics **only from the training data**, then apply them to val/test, exactly following the logic of the previous Tehran price project.
- Text features (`Neighborhood (English)` in Source, `Address` in Target) are encoded with `pd.get_dummies`.
- `target.py` also has one extra feature that `source.py` does not: **`facility_score`** (the sum of rooms, parking, elevator, and storage), since this information is not available in the Source dataset.

## 🧠 Model Architecture

### Independent Full Models (inside `source.py` and `target.py`)

Both use a similar 4-layer network (`Linear(52) → ReLU → Linear(94) → ReLU → Linear(258) → ReLU → Linear(output)`), with `L1Loss`, Early Stopping, and checkpointing of the best weights (`copy.deepcopy`/`load_state_dict`), trained on **all available features** of each dataset (not just the 4 shared features).

### Transfer Learning Model (in `main.py`)

A separate class named `MLP4` is defined with a **fixed input of 4 features** (`Base Area`/`Area`, `district_number`, `is_suburb`, `top_area`). It has the same 4-layer architecture, but since the inputs of both datasets must be the same size for layers to be swapped between them, only this shared subset is used.

**Transfer process:**
1. `fc1` and `fc2` of the Transfer model are copied from the **Source** model (layers that learned general, broader patterns of housing prices).
2. `fc3` and `fc4` are copied from the **Target baseline** model.
3. **Stage 1**: `fc1`/`fc2` are frozen (`requires_grad=False`) and only `fc3`/`fc4` are trained on the Target data. In other words, the model learns to "translate" the Source representations for the Target problem.
4. **Stage 2**: All layers are unfrozen and the entire network is fine-tuned with a much smaller learning rate (`5e-5`), so that it fully adapts to the Target data without completely losing the transferred knowledge.

## 📊 Datasets

| | Source | Target |
| --- | --- | --- |
| File | `en_data.csv` | `tehranHprice.csv` |
| Number of records (4 shared features) | 139,749 | 3,473 |
| Target variable | `Total Price` (with `log1p`) | `Price` (with `log1p`) |
| Extra features (only in this dataset) | — | `facility_score` |

## ⚙️ Installing Prerequisites

```bash
pip install pandas numpy scikit-learn torch
```

## 🚀 How to Run

Place the `en_data.csv` and `tehranHprice.csv` files in the same directory as the scripts, then run:

```bash
python main.py
```

As explained in the "Project Structure" section, this command automatically runs `source.py` and `target.py` as well, so the output will include all four sections (the full Source model, the full Target model, and the Transfer Learning stages).

## 📈 Results

### Independent Full Models (with all features)

| Model | Train R² | Test R² | Test MAE | Test RMSE |
| --- | --- | --- | --- | --- |
| Source (full) | 0.934 | 0.919 | 72,627,960 | 192,154,820 |
| Target (full) | 0.816 | 0.840 | 1,116,639,600 | 3,724,102,100 |

> 📌 The scale of the MAE/RMSE values is not directly comparable between the two datasets, because their price ranges differ; R² is a unitless metric that gives a fairer comparison.

### Transfer Learning Process (with only the 4 shared features)

| Stage | R² | MAE | RMSE |
| --- | --- | --- | --- |
| Source (4 features) | 0.8373 | 118,517,247 | 272,181,551 |
| **Target Baseline** (4 features, no transfer) | **0.8505** | 1,390,623,366 | 2,747,342,057 |
| Transfer — Stage 1 (Source layers frozen) | 0.8399 | 1,441,039,540 | 2,842,905,927 |
| **Transfer — Stage 2 (full fine-tuning)** | **0.8663** | 1,326,050,056 | 2,598,063,340 |

### Final Effect of Knowledge Transfer (Transfer Effect)

| Metric | Change relative to Baseline |
| --- | --- |
| Δ R² | **+0.0158** |
| Δ MAE | −64,573,310 |
| Δ RMSE | −149,278,716 |

**✅ Result: Positive Transfer.** The model that used knowledge from the Source dataset has both a higher R² and lower error compared with the baseline model trained only on the Target data (limited to 4 features).

> 📌 Another interesting note: the Target baseline model (with only 4 features, R²=0.8505) performed slightly better even than the full Target model (with all features, including the One-Hot encoded addresses, R²=0.840). This is most likely due to the lack of data (only 3,473 records) relative to the large number of One-Hot address categories; with such limited data, sparse features can cause overfitting, whereas 4 compact, meaningful features generalize better.

## 🛠️ Technologies

- Python
- Pandas / NumPy
- Scikit-learn (MinMaxScaler, train_test_split, regression metrics)
- PyTorch

## 📝 Additional Notes

- The different `top_area` condition between `source.py` and `target.py` (District 1 alone versus Districts 1–3) shows that this feature was chosen based on independent experimentation and empirical analysis on each dataset, not as a uniform rule copied between projects.
- The reason for choosing `fc1`/`fc2` (rather than the final layers) for transfer from Source is that the early layers of a network typically learn more general, transferable patterns, while the last layers are more specific to the particular problem (the Target dataset). For this reason, the final layers were taken from the Target baseline model.
- The learning rate of fine-tuning Stage 2 (`5e-5`) is intentionally much smaller than that of Stage 1 and of the initial training, so that the transferred weights are not abruptly destroyed and the fine-tuning proceeds gently.
- Both independent scripts (`source.py`, `target.py`) and the main script (`main.py`) use the standard pattern of these projects: a three-way train/val/test split, Early Stopping based on val (not test), and restoring the best weights with `copy.deepcopy`.

## 📄 License

This project was created for educational/personal purposes.
