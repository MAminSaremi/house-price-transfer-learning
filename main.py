import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#? Source and Target files for two purposes:
#? 1. Transferring four shared features as the input X
#? 2. Transferring the corresponding Y for each dataset
import source
import target


#! ============================================================
#! SETTINGS
#! ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


#! ============================================================
#! 1. GET THE 4 SHARED FEATURES
#! ============================================================
#? These two datasets have the same task and similar domains, but they do not have the same number of features.
#? Therefore, to make the transfer practical, I selected four shared and similar features from both datasets as 
#? Source X and Target X.
# In source.py:
# transfer_x = df[
#     ['Base Area', 'district_number', 'is_suburb', 'top_area']
# ]

# In target.py:
# transfer_x = df[
#     ['Area', 'district_number', 'is_suburb', 'top_area']
# ]

source_x = source.transfer_x.copy()
target_x = target.transfer_x.copy()

source_y = source.y.copy()
target_y = target.y.copy()


# Convert to numpy
source_x = source_x.astype(np.float32).values
target_x = target_x.astype(np.float32).values

source_y = np.asarray(source_y).reshape(-1, 1)
target_y = np.asarray(target_y).reshape(-1, 1)


print("\nSource shape:", source_x.shape)
print("Target shape:", target_x.shape)


#! ============================================================
#! 2. SPLIT SOURCE
#!    60% TRAIN
#!    20% VALIDATION
#!    20% TEST
#! ============================================================

source_train_x, source_temp_x, \
source_train_y, source_temp_y = train_test_split(
    source_x,
    source_y,
    test_size=0.40,
    random_state=SEED
)

source_val_x, source_test_x, \
source_val_y, source_test_y = train_test_split(
    source_temp_x,
    source_temp_y,
    test_size=0.50,
    random_state=SEED
)


#! ============================================================
#! 3. SPLIT TARGET
#!    80% TRAIN
#!    10% VALIDATION
#!    10% TEST
#! ============================================================

target_train_x, target_temp_x, \
target_train_y, target_temp_y = train_test_split(
    target_x,
    target_y,
    test_size=0.20,
    random_state=SEED
)

target_val_x, target_test_x, \
target_val_y, target_test_y = train_test_split(
    target_temp_x,
    target_temp_y,
    test_size=0.50,
    random_state=SEED
)


#! ============================================================
#! 4. SCALE X
#! ============================================================

# Source scaler
source_x_scaler = MinMaxScaler()

source_train_x = source_x_scaler.fit_transform(source_train_x)
source_val_x = source_x_scaler.transform(source_val_x)
source_test_x = source_x_scaler.transform(source_test_x)


#? Target transfer data is transformed using SOURCE scaler.
#? This is deliberate.
#? We want to see whether the Source representation transfers.
target_train_x_transfer = source_x_scaler.transform(target_train_x)
target_val_x_transfer = source_x_scaler.transform(target_val_x)
target_test_x_transfer = source_x_scaler.transform(target_test_x)


#* ------------------------------------------------------------
#* Target-only scaler
#*
#* This is used ONLY for the Target baseline.
#* ------------------------------------------------------------

target_x_scaler = MinMaxScaler()

target_train_x_baseline = target_x_scaler.fit_transform(target_train_x)
target_val_x_baseline = target_x_scaler.transform(target_val_x)
target_test_x_baseline = target_x_scaler.transform(target_test_x)


#! ============================================================
#! 5. SCALE Y
#! ============================================================

source_y_scaler = MinMaxScaler()

source_train_y = source_y_scaler.fit_transform(source_train_y)
source_val_y = source_y_scaler.transform(source_val_y)
source_test_y = source_y_scaler.transform(source_test_y)


target_y_scaler = MinMaxScaler()

target_train_y = target_y_scaler.fit_transform(target_train_y)
target_val_y = target_y_scaler.transform(target_val_y)
target_test_y = target_y_scaler.transform(target_test_y)


#! ============================================================
#! 6. MLP
#! ============================================================

class MLP4(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(4, 52)

        self.fc2 = nn.Linear(52, 94)

        self.fc3 = nn.Linear(94, 258)

        self.fc4 = nn.Linear(258, 1)


    def forward(self, x):

        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        x = torch.relu(self.fc3(x))

        x = self.fc4(x)

        return x


#! ============================================================
#! 7. DATALOADER
#! ============================================================

def create_loader(x, y, batch_size):

    x = torch.tensor(
        x,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.float32
    )

    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader



#! ============================================================
#! 8. TRAIN FUNCTION
#! ============================================================

def train_model( model, train_x, train_y, val_x, val_y, lr, batch_size, epochs, patience, freeze_fc12=False ):

    model = model.to(DEVICE)


    # --------------------------------------------------------
    # Freeze Source layers if requested
    # --------------------------------------------------------

    if freeze_fc12:

        for param in model.fc1.parameters():
            param.requires_grad = False

        for param in model.fc2.parameters():
            param.requires_grad = False

    else:

        for param in model.parameters():
            param.requires_grad = True


    optimizer = torch.optim.Adam(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=lr
    )


    criterion = nn.L1Loss()


    train_loader = create_loader(
        train_x,
        train_y,
        batch_size
    )


    val_x_tensor = torch.tensor(
        val_x,
        dtype=torch.float32
    ).to(DEVICE)

    val_y_tensor = torch.tensor(
        val_y,
        dtype=torch.float32
    ).to(DEVICE)


    best_loss = float("inf")

    best_state = copy.deepcopy(
        model.state_dict()
    )

    patience_counter = 0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(epochs):

        model.train()


        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)

            batch_y = batch_y.to(DEVICE)


            optimizer.zero_grad()


            prediction = model(batch_x)


            loss = criterion(
                prediction,
                batch_y
            )


            loss.backward()


            optimizer.step()


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_prediction = model(
                val_x_tensor
            )

            val_loss = criterion(
                val_prediction,
                val_y_tensor
            ).item()


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= patience:

            print(
                f"Early stopping at epoch {epoch + 1}"
            )

            break


    # Restore best model

    model.load_state_dict(best_state)


    return model


#! ============================================================
#! 9. EVALUATION
#! ============================================================

def evaluate_model(model,test_x,test_y,y_scaler):

    model.eval()


    test_x_tensor = torch.tensor(
        test_x,
        dtype=torch.float32
    ).to(DEVICE)


    with torch.no_grad():

        prediction = model(
            test_x_tensor
        ).cpu().numpy()


    # Reverse MinMaxScaler

    prediction_log = y_scaler.inverse_transform(
        prediction
    ).reshape(-1)


    true_log = y_scaler.inverse_transform(
        test_y
    ).reshape(-1)


    # Reverse log1p

    prediction_price = np.expm1(
        prediction_log
    )

    true_price = np.expm1(
        true_log
    )


    r2 = r2_score(
        true_price,
        prediction_price
    )

    mae = mean_absolute_error(
        true_price,
        prediction_price
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_price,
            prediction_price
        )
    )


    return r2, mae, rmse


#! ============================================================
#! 10. SOURCE MODEL
#! ============================================================

print("\n")
print("=" * 60)
print("SOURCE 4-FEATURE MODEL")
print("=" * 60)


source_model = MLP4()


source_model = train_model(

    source_model,

    source_train_x,
    source_train_y,

    source_val_x,
    source_val_y,

    lr=0.001,

    batch_size=128,

    epochs=500,

    patience=50
)


source_result = evaluate_model(

    source_model,

    source_test_x,
    source_test_y,

    source_y_scaler
)


print(
    f"Source Test R2   : {source_result[0]:.4f}"
)

print(
    f"Source Test MAE  : {source_result[1]:,.0f}"
)

print(
    f"Source Test RMSE : {source_result[2]:,.0f}"
)


#! ============================================================
#! 11. TARGET BASELINE
#! ============================================================

print("\n")
print("=" * 60)
print("TARGET 4-FEATURE BASELINE")
print("=" * 60)


target_model = MLP4()


target_model = train_model(

    target_model,

    target_train_x_baseline,
    target_train_y,

    target_val_x_baseline,
    target_val_y,

    lr=0.0001,

    batch_size=50,

    epochs=500,

    patience=100
)


target_result = evaluate_model(

    target_model,

    target_test_x_baseline,
    target_test_y,

    target_y_scaler
)


print(
    f"Target Baseline R2   : {target_result[0]:.4f}"
)

print(
    f"Target Baseline MAE  : {target_result[1]:,.0f}"
)

print(
    f"Target Baseline RMSE : {target_result[2]:,.0f}"
)


#! ============================================================
#! 12. CREATE TRANSFER MODEL
#! ============================================================

print("\n")
print("=" * 60)
print("TRANSFER MODEL")
print("=" * 60)


transfer_model = MLP4()


# ------------------------------------------------------------
# SOURCE -> FC1
# ------------------------------------------------------------

transfer_model.fc1.load_state_dict(

    copy.deepcopy(
        source_model.fc1.state_dict()
    )
)


# ------------------------------------------------------------
# SOURCE -> FC2
# ------------------------------------------------------------

transfer_model.fc2.load_state_dict(

    copy.deepcopy(
        source_model.fc2.state_dict()
    )
)


# ------------------------------------------------------------
# TARGET -> FC3
# ------------------------------------------------------------

transfer_model.fc3.load_state_dict(

    copy.deepcopy(
        target_model.fc3.state_dict()
    )
)


# ------------------------------------------------------------
# TARGET -> FC4
# ------------------------------------------------------------

transfer_model.fc4.load_state_dict(

    copy.deepcopy(
        target_model.fc4.state_dict()
    )
)


print("\nTransferred architecture:")

print(
    "fc1 = SOURCE"
)

print(
    "fc2 = SOURCE"
)

print(
    "fc3 = TARGET"
)

print(
    "fc4 = TARGET"
)


#! ============================================================
#! 13. TRANSFER STAGE 1
#!
#! Freeze Source fc1 + fc2
#! Train only Target fc3 + fc4
#! ============================================================

print("\n")
print("-" * 60)
print("TRANSFER STAGE 1")
print("Source fc1/fc2 frozen")
print("-" * 60)


transfer_model = train_model(

    transfer_model,

    target_train_x_transfer,
    target_train_y,

    target_val_x_transfer,
    target_val_y,

    lr=0.0001,

    batch_size=50,

    epochs=100,

    patience=25,

    freeze_fc12=True
)


stage1_result = evaluate_model(

    transfer_model,

    target_test_x_transfer,
    target_test_y,

    target_y_scaler
)


print(
    f"Stage 1 R2   : {stage1_result[0]:.4f}"
)

print(
    f"Stage 1 MAE  : {stage1_result[1]:,.0f}"
)

print(
    f"Stage 1 RMSE : {stage1_result[2]:,.0f}"
)


#! ============================================================
#! 14. TRANSFER STAGE 2
#!
#! Unfreeze everything
#! Fine-tune the whole network
#! ============================================================

print("\n")
print("-" * 60)
print("TRANSFER STAGE 2")
print("All layers unfrozen")
print("-" * 60)


transfer_model = train_model(

    transfer_model,

    target_train_x_transfer,
    target_train_y,

    target_val_x_transfer,
    target_val_y,

    lr=0.00005,

    batch_size=50,

    epochs=300,

    patience=50,

    freeze_fc12=False
)


transfer_result = evaluate_model(

    transfer_model,

    target_test_x_transfer,
    target_test_y,

    target_y_scaler
)


print(
    f"Transfer R2   : {transfer_result[0]:.4f}"
)

print(
    f"Transfer MAE  : {transfer_result[1]:,.0f}"
)

print(
    f"Transfer RMSE : {transfer_result[2]:,.0f}"
)


#! ============================================================
#! 15. FINAL COMPARISON
#! ============================================================

print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)


print("\nTarget 4-feature baseline:")

print(
    f"R2   = {target_result[0]:.4f}"
)

print(
    f"MAE  = {target_result[1]:,.0f}"
)

print(
    f"RMSE = {target_result[2]:,.0f}"
)


print("\nTransfer:")

print(
    f"R2   = {transfer_result[0]:.4f}"
)

print(
    f"MAE  = {transfer_result[1]:,.0f}"
)

print(
    f"RMSE = {transfer_result[2]:,.0f}"
)


#! ============================================================
#! 16. TRANSFER EFFECT
#! ============================================================

delta_r2 = (
    transfer_result[0]
    -
    target_result[0]
)


delta_mae = (
    transfer_result[1]
    -
    target_result[1]
)


delta_rmse = (
    transfer_result[2]
    -
    target_result[2]
)


print("\n")
print("=" * 60)
print("TRANSFER EFFECT")
print("=" * 60)


print(
    f"Δ R2   = {delta_r2:+.4f}"
)

print(
    f"Δ MAE  = {delta_mae:+,.0f}"
)

print(
    f"Δ RMSE = {delta_rmse:+,.0f}"
)


if delta_r2 > 0:

    print(
        "\n Positive transfer!"
    )

else:

    print(
        "\n Negative transfer / no improvement."
    )



# Epoch: 25 | Train Loss: 0.0176 | Val Loss: 0.0181
# Epoch: 50 | Train Loss: 0.0171 | Val Loss: 0.0181
# Epoch: 75 | Train Loss: 0.0168 | Val Loss: 0.0178
# Epoch: 100 | Train Loss: 0.0166 | Val Loss: 0.0178
# Epoch: 125 | Train Loss: 0.0164 | Val Loss: 0.0179
# Epoch: 150 | Train Loss: 0.0163 | Val Loss: 0.0177
# Epoch: 175 | Train Loss: 0.0162 | Val Loss: 0.0178
# Epoch: 200 | Train Loss: 0.0161 | Val Loss: 0.0177
# Epoch: 225 | Train Loss: 0.0160 | Val Loss: 0.0179
# Early stopping at epoch 237
# Training MAE  - 64899690.0
# Training RMSE - 164922060.0
# Training R²   - 0.934

# Test MAE  - 72627960.0
# Test RMSE - 192154820.0
# Test R²   - 0.919
# Epoch: 25 | Train Loss: 0.0230 | Val Loss: 0.0242
# Epoch: 50 | Train Loss: 0.0206 | Val Loss: 0.0228
# Epoch: 75 | Train Loss: 0.0195 | Val Loss: 0.0216
# Epoch: 100 | Train Loss: 0.0182 | Val Loss: 0.0217
# Epoch: 125 | Train Loss: 0.0174 | Val Loss: 0.0208
# Epoch: 150 | Train Loss: 0.0167 | Val Loss: 0.0219
# Epoch: 175 | Train Loss: 0.0162 | Val Loss: 0.0204
# Epoch: 200 | Train Loss: 0.0158 | Val Loss: 0.0207
# Epoch: 225 | Train Loss: 0.0161 | Val Loss: 0.0214
# Epoch: 250 | Train Loss: 0.0154 | Val Loss: 0.0207
# Epoch: 275 | Train Loss: 0.0154 | Val Loss: 0.0208
# Epoch: 300 | Train Loss: 0.0151 | Val Loss: 0.0206
# Early stopping at epoch 315
# Training MAE  - 942730940.0
# Training RMSE - 3459510500.0
# Training R²   - 0.816

# Test MAE  - 1116639600.0
# Test RMSE - 3724102100.0
# Test R²   - 0.84
# Device: cpu

# Source shape: (139749, 4)
# Target shape: (3473, 4)


# ============================================================
# SOURCE 4-FEATURE MODEL
# ============================================================
# Early stopping at epoch 93
# Source Test R2   : 0.8373
# Source Test MAE  : 118,517,247
# Source Test RMSE : 272,181,551


# ============================================================
# TARGET 4-FEATURE BASELINE
# ============================================================
# Target Baseline R2   : 0.8505
# Target Baseline MAE  : 1,390,623,366
# Target Baseline RMSE : 2,747,342,057


# ============================================================
# TRANSFER MODEL
# ============================================================

# Transferred architecture:
# fc1 = SOURCE
# fc2 = SOURCE
# fc3 = TARGET
# fc4 = TARGET


# ------------------------------------------------------------
# TRANSFER STAGE 1
# Source fc1/fc2 frozen
# ------------------------------------------------------------
# Stage 1 R2   : 0.8399
# Stage 1 MAE  : 1,441,039,540
# Stage 1 RMSE : 2,842,905,927


# ------------------------------------------------------------
# TRANSFER STAGE 2
# All layers unfrozen
# ------------------------------------------------------------
# Early stopping at epoch 280
# Transfer R2   : 0.8663
# Transfer MAE  : 1,326,050,056
# Transfer RMSE : 2,598,063,340


# ============================================================
# FINAL COMPARISON
# ============================================================

# Target 4-feature baseline:
# R2   = 0.8505
# MAE  = 1,390,623,366
# RMSE = 2,747,342,057

# Transfer:
# R2   = 0.8663
# MAE  = 1,326,050,056
# RMSE = 2,598,063,340


# ============================================================
# TRANSFER EFFECT
# ============================================================
# Δ R2   = +0.0158
# Δ MAE  = -64,573,310
# Δ RMSE = -149,278,716

# ✅ Positive transfer!