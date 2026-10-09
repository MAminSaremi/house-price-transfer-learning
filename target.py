import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader , TensorDataset
from torch import optim
import copy
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

df = pd.read_csv("tehranHprice.csv")

# print(df['Address'].unique())

#? The original dataset does not include the official district number for each address.
#? However, in the real Tehran housing market, the district is one of the strongest factors
#? affecting property prices. Therefore, I manually mapped each address to its corresponding
#? district and added it as a new feature. Additionally, addresses located in suburban areas
#? around Tehran were identified and added as a separate binary feature.

neighborhood_to_district = {
    # منطقه ۱
    'Velenjak': 1, 'Zaferanieh': 1, 'Niavaran': 1, 'Kamranieh': 1, 'Aqdasieh': 1,'Sohanak':1,
    'Darband': 1, 'Darakeh': 1, 'Tajrish': 1, 'Farmanieh': 1, 'Ozgol': 1,'Gheitarieh':1,
    'Araj': 1, 'Dezashib': 1, 'Ekhtiarieh': 1, 'Hekmat': 1,'Mahallati':1,'Elahieh':1,
    'Mahmoudieh':1,'Ajudaniye':1,'Chidz':1,

    # منطقه ۲
    'Shahrake Gharb': 2, 'Marzdaran': 2, 'Punak': 2,'Sadeghieh': 2,'Shahrake Qods':2,'Saadat Abad':2,'ShahrAra':2,'Gisha':2,
    'Sattarkhan': 2,'Daryan No':2,'Tarasht':2,'Telecommunication':2,'Shahrake Quds':2,

    # منطقه ۳
    'Vanak': 3, 'Jordan': 3, 'Mirdamad': 3, 
    'Zafar': 3, 'Gholhak': 3, 'Zargandeh': 3,'Dorous': 3,
    'Pasdaran':3,'Mirdamad':3,'Jordan':3,
    'Vanak':3,'Ghoba':3,'Seyed Khandan':3,

    # منطقه ۴
    'Lavizan': 4, 'Hakimiyeh': 4,'West Pars':4,'Shams Abad':4,'Elm-o-Sanat':4,
    'Heravi':4,'East Pars':4,'Mehran':4,'Kazemabad':4,

    # منطقه ۵
    'Ekbatan': 5,'Shahran':5,'Andisheh':5,'Feiz Garden':5,'Water Organization':5,'Chardivari':5,'Qalandari':5,'Abazar':5,'Shahrakeh Naft':5,
    'Koohsar':5,'Shahrake Apadana': 5,'Shahr-e-Ziba': 5,'West Ferdows Boulevard': 5, 'East Ferdows Boulevard': 5,'Central Janatabad': 5, 'Northern Janatabad': 5, 'Southern Janatabad': 5,
    'Eram':5,'North Program Organization':5,'Southern Program Organization':5,

    # منطقه ۶
    'Yousef Abad': 6, 'Amirabad': 6, 'Keshavarz Boulevard': 6, 'Fatemi': 6,'Mirza Shirazi':6,'Argentina':6,
    'Karimkhan': 6,'Valiasr':6,'Amirabad':6,'Villa':6,'Yousef Abad':6,'Gandhi':6,

    # منطقه ۷
    'Abbasabad': 7, 'Nezamabad': 7, 'Heshmatieh': 7,'Bahar':7,'Haft Tir':7,'Northern Suhrawardi':7,'Garden of Saba':7,
    
    # 9
    'Ostad Moein' : 9,'Si Metri Ji':9,
    # منطقه ۸
    'Narmak': 8, 'Majidieh': 8, 'Tehran Now': 8, 'Sabalan': 8,'Vahidieh':8,'Shahrake Madaen':8,'Taslihat':8,
    'Vahidiyeh':8,

    # منطقه ۱۰
    'Beryanak': 10,'Salsabil':10,'Qasr-od-Dasht':10,'Jeyhoon':10,'Hashemi':10,'Karoon':10,'Komeil':10,
    'Nawab':10,

    # منطقه ۱۱
    'Enghelab': 11, 'Republic': 11, 'Amirieh': 11,'Moniriyeh' :11,'Northren Jamalzadeh':11,'Azarbaijan':11,'Amir Bahador':11,
    'Razi':11,'Eskandari':11,

    # منطقه ۱۲
    'Waterfall':12,'Hassan Abad':12,

    # منطقه ۱۳
    'Pirouzi': 13,'Air force':13,'Thirteen November':13,
    # 14
    'Parastar':14,'Ahang':14,

    # منطقه ۱۵
    'Afsarieh': 15, 'Khavaran': 15,'Atabak':15,

    # منطقه ۱۶
    'Naziabad': 16,'Railway':16,'Shoosh':16,'Aliabad South':16,'Javadiyeh':16,'Yakhchiabad':16,
    # 17
    'Qazvin Imamzadeh Hassan':17,'Fallah':17,'Azari':17,'Boloorsazi':17,
    # 18
    'Yaftabad':18,'Shadabad':18,
    # 19
    'Baghestan':19,'Salehabad':19,
    # منطقه ۲۰
    'Ray': 20, 'Ray - Montazeri': 20, 'Ray - Pilgosh': 20,
    # 21
    'Tehransar':21,'Shahrake Azadi':21,
    # منطقه ۲۲
    'Northern Chitgar': 22, 'Southern Chitgar': 22, 'Dehkade Olampic': 22,'Shahrake Shahid Bagheri':22,'Persian Gulf Martyrs Lake':22
    ,'Golestan':22,'Southern Chitgar':22,'Kook':22,'Azadshahr':22,'Zibadasht':22
}
df['district_number'] = df['Address'].map(neighborhood_to_district)  

suburb={
    'Pardis',
    'Islamshahr',
    'Parand',
    'Pakdasht',
    'Pakdasht KhatunAbad',
    'Shahryar',
    'Rudhen',
    'Chahardangeh',
    'Baqershahr',
    'Kahrizak',
    'Qarchak',
    'Damavand',
    'Absard',
    'Lavasan',
    'Shahedshahr',
    'Nasim Shahr',
    'Chardangeh',
    'Tenant',
    'Malard',
    'SabaShahr',
    'Pishva',
    'Islamshahr Elahieh',
    'Ray - Montazeri',
    'Firoozkooh Kuhsar',
    'Robat Karim',
    'Ray - Pilgosh',
    'Ghiyamdasht',
    'Safadasht',
    'Khademabad Garden',
    'Mehrabad River River',
    'Varamin - Beheshti',
    'Alborz Complex',
    'Firoozkooh'
}
pattern = '|'.join(suburb)
df['is_suburb'] = df['Address'].str.contains(pattern,case=False, na=False)

#? EDA revealed a premium housing segment characterized by:
#? - Area > 120 square meters
#? - Located in districts 1, 2, or 3
#?
#? Properties satisfying these conditions showed a substantial price gap compared to
#? the remaining observations. To help the model explicitly learn this nonlinear
#? market behavior, a dedicated feature was engineered to identify this premium group.
df["top_area"] = False
mask = (
    (df["Area"] > 120) &
    (df["district_number"].isin([1, 2, 3]))
)
df.loc[mask, "top_area"] = True

#? To help the model better capture the effect of property amenities on price,
#? a facility_score feature was created by summing the availability of key
#? amenities (Room,parking, elevator, and warehouse). This provides a simple
#? numerical representation of the overall facility level of each property.
df['facility_score'] = 0
facility_score = (
    df["Room"] +
    df["Parking"].astype(int) +
    df["Elevator"].astype(int) +
    df["Warehouse"].astype(int)
)
df['facility_score'] = facility_score





# plt.figure(figsize=(12, 5))
# plt.bar(district_stats.index.astype(str), district_stats['mean'])
# plt.xlabel('شماره منطقه')
# plt.ylabel('میانگین قیمت')
# plt.title('میانگین قیمت به تفکیک منطقه')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()


#? WE HAVE 23 MISSINGVALUES IN ADDRESS COLUMN
# print(df.loc[df['Address'].isna()])
df['Address'] = df['Address'].fillna('missing')

df = df[df["Price"] > 0]



#? District numbers are categorical identifiers rather than numerical values.
#? Missing values are replaced with "0", then converted to integers to remove
#? decimal representations (e.g., 2.0 -> 2), and finally converted to strings
#? so they are treated as categorical values instead of continuous numerical features.
df["district_number"] = (
    df["district_number"]
      .fillna(0)
      .astype(int)
      .astype(str)
)

#? To prepare the text-based Address feature for the neural network,
#? I applied one-hot encoding using get_dummies.
ch_cols = ['Address']
df = pd.get_dummies(df , columns=ch_cols)

x = df.drop(columns=["Price" , "Price(USD)"])


# !print((df["Price"] / df["Price(USD)"]).describe())
# !
# !print(df[["Price", "Price(USD)"]].corr())

y = np.log1p(df["Price"])


train_x , temp_x , train_y , temp_y = train_test_split(x , y , test_size=0.2 , random_state=42)
test_x , val_x , test_y , val_y = train_test_split(temp_x , temp_y , test_size=0.5 , random_state=42)

#? Using target encoding, I calculated the average price for each district
#? in a way that prevents data leakage.
#? I used Smoothed Target Encoding to reduce the effect of districts
#? with a small number of samples.

def target_encoding(X_train, X_val, X_test, y_train, m=50):

    train_df = X_train.copy()
    train_df["target"] = y_train

    global_mean = train_df["target"].mean()

    stats = train_df.groupby("district_number")["target"].agg(
        ["mean", "count"]
    )

    stats["smooth"] = (
        stats["count"] * stats["mean"] +
        m * global_mean
    ) / (stats["count"] + m)

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train["district_target_mean"] = (
        X_train["district_number"].map(stats["smooth"])
    )

    X_val["district_target_mean"] = (
        X_val["district_number"].map(stats["smooth"])
    )

    X_test["district_target_mean"] = (
        X_test["district_number"].map(stats["smooth"])
    )

    X_train["district_target_mean"] = (
        X_train["district_target_mean"].fillna(global_mean)
    )

    X_val["district_target_mean"] = (
        X_val["district_target_mean"].fillna(global_mean)
    )

    X_test["district_target_mean"] = (
        X_test["district_target_mean"].fillna(global_mean)
    )

    return X_train, X_val, X_test

train_x, val_x, test_x = target_encoding(
    train_x,
    val_x,
    test_x,
    train_y
)

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
y_scaler= MinMaxScaler()
train_x = x_scaler.fit_transform(train_x)
test_x = x_scaler.transform(test_x)
val_x = x_scaler.transform(val_x)

train_y = y_scaler.fit_transform(train_y.values.reshape(-1,1))
test_y = y_scaler.transform(test_y.values.reshape(-1,1))
val_y = y_scaler.transform(val_y.values.reshape(-1,1))

train_x = torch.tensor(train_x,dtype=torch.float32)
test_x = torch.tensor(test_x,dtype=torch.float32)
val_x = torch.tensor(val_x,dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32)

epochs = 500
batch_size = 50
loss_function = nn.L1Loss()

class neuralnetwork2(nn.Module):
    def __init__(self,input_dim = train_x.shape[1] , output_dim = train_y.shape[1]):
        super().__init__()
        torch.manual_seed(2026)
        self.fc1 = nn.Linear(input_dim, 52)
        self.fc2 = nn.Linear(52,94)
        self.fc3 = nn.Linear(94,258)
        self.fc4 = nn.Linear(258,output_dim)
        self.relu = nn.ReLU()
        
    def forward(self , x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        
        return x
model = neuralnetwork2()

lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
train_data = TensorDataset(train_x ,train_y)
train_loader = DataLoader(train_data  , batch_size=batch_size , shuffle=True)


def train_network(model, loss_function, optimizer, epochs, train_loader):
    total_loss=[]
    counter=0
    patience = 100
    best_val_loss = float('inf')
    best_model_state= copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batch=0

        for batch_x , batch_y in train_loader:
            output_data = model(batch_x)
            loss = loss_function(output_data,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss +=loss.item()
            num_batch +=1
        epoch_loss = train_loss/num_batch
        total_loss.append(epoch_loss)

        model.eval()
        with torch.inference_mode():
            test_output = model(val_x)
            test_loss = loss_function(test_output,val_y)
        if test_loss.item()< best_val_loss :
            best_val_loss = test_loss.item()
            counter=0
            best_model_state = copy.deepcopy(model.state_dict())
        else :
            counter+=1
            if counter>=patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        if (epoch + 1) % 25 == 0:
                                    
                                    print(
                                        f"Epoch: {epoch + 1} | "
                                        f"Train Loss: {epoch_loss:.4f} | "
                                        f"Val Loss: {test_loss.item():.4f}"
                                    )
    model.load_state_dict(best_model_state)
    return total_loss

def eval_model(model,train_x , train_y , test_x , test_y , scaler_y):
      model.eval()
      with torch.inference_mode():
            train_pred = model(train_x)
            test_pred = model(test_x)

            train_pred = train_pred.numpy()
            test_pred = test_pred.numpy()
            train_y = train_y.numpy()
            test_y=test_y.numpy()

            train_pred  = scaler_y.inverse_transform(train_pred)
            test_pred  = scaler_y.inverse_transform(test_pred)
            train_y  = scaler_y.inverse_transform(train_y)
            test_y  = scaler_y.inverse_transform(test_y)

            train_pred = np.expm1(train_pred)
            test_pred = np.expm1(test_pred)
            train_y = np.expm1(train_y)
            test_y = np.expm1(test_y)

            train_mae = mean_absolute_error(train_y, train_pred)
            train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
            train_r2 = r2_score(train_y, train_pred)
        
            print("Training MAE  -", round(train_mae, 2))
            print("Training RMSE -", round(train_rmse, 2))
            print("Training R²   -", round(train_r2, 3))
        
            print()
        
            # -------------------------
            # Test Metrics
            # -------------------------
        
            test_mae = mean_absolute_error(test_y, test_pred)
            test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
            test_r2 = r2_score(test_y, test_pred)
        
            print("Test MAE  -", round(test_mae, 2))
            print("Test RMSE -", round(test_rmse, 2))
            print("Test R²   -", round(test_r2, 3))
           
train_network(
    model,
    loss_function,
    optimizer,
    epochs,
    train_loader
)
            
eval_model(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    y_scaler
)
#? Shared features used for Transfer Learning
transfer_x = df[['Area','district_number','is_suburb','top_area']]
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