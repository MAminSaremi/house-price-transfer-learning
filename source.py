import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch




df = pd.read_csv('en_data.csv')


df = df.drop(columns=['Submission Date','Exact Location','Price per Square Meter', 'Neighborhood Name'])

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
    'Mahmoudieh':1,'Ajudaniye':1,'Chidz':1,'Piche Shemran':1,'Ajodanie':1,'Aghdasieh':1,
    'Mini City':1,'Sohanank':1,'Gheitarie':1,'Evin':1,'Artesh (Lashkarak)':1,'Fereshteh':1,
    'Mahmoodieh':1,
    'دارآباد':1,'ازگل':1,'دزاشیب':1, 'شمال میدان Tajrish':1,'نوبنیاد':1,'شریعتی(از Tajrish تا پل صدر)':1,'باغ فردوس':1,

    # منطقه ۲
    'Shahrake Gharb': 2, 'Marzdaran': 2, 'Poonak': 2,'Sadeghieh': 2,'Shahrake Qods':2,'Saadat Abad':2,'ShahrAra':2,'Gisha':2,
    'Sattarkhan': 2,'Daryan No':2,'Tarasht':2,'Telecommunication':2,'Shahrake Quds':2,'Ashrafi Esfahani (Bolvar ta Hakim)':2,
    'Ashrafi Esfahan (Sadeghieh ta Hakim)':2,'Kooye Faraz':2,'Tehran Vila':2,'Mazandaran':2,'Shahrak Gharb':2,'Shahr Ara':2,
    'بلوار آسیا':2,'فرحزاد':2,

    # منطقه ۳
    'Vanak': 3, 'Jordan': 3, 'Mirdamad': 3, 'Jolfa':3,
    'Zafar': 3, 'Gholhak': 3, 'Zargandeh': 3,'Dorous': 3,
    'Pasdaran':3,'Mirdamad':3,'Jordan':3,'Araghi':3,'Seul':3,
    'Vanak':3,'Ghoba':3,'Seyed Khandan':3,'Khaje Abdollah':3,
    'Sheikh Bahaii':3,'Dibaji Jonoobi':3,'Shariati (Shiraz-Shemran)':3,
    'Dibaji Shomali':3,'Dolat':3,
    'ظفر':3,'میرداماد':3,'شریعتی(از صدر تا همت)':3,'ملاصدرا':3, 'ونک':3,'سیزده آبان':3,'بخارست':3,
    'شریعتی( ازهمت تا بهارشیراز)':3,'ولیعصر( از ونک تا پارک وی)':3,'سید خندان':3,'Valiasr (Park way)':3,

    # منطقه ۴
    'Lavizan': 4, 'Hakimieh': 4,'Tehranpars':4,'Shams Abad':4,'Elm-o-Sanat':4,'Farjam Gharbi':4,'Police':4,
    'Heravi':4,'East Pars':4,'Mehran':4,'Kazemabad':4,'Majidieh Shomali':4,'Jashnvareh':4,'Ghanat Kosar':4,
    'Hengam':4,'Delavaran':4,'Daroos':4,'Bani Hashem':4,'Seraj':4,'Lavisan':4,
    'میدان رسالت':4,'شهرک امید':4,'Darou':4,'دیلمان':4,

    # منطقه ۵
    'Ekbatan': 5,'Shahran':5,'Andisheh':5,'Feiz Garden':5,'Water Organization':5,'Chardivari':5,'Qalandari':5,'Abazar':5,'Shahrakeh Naft':5,
    'Koohsar':5,'Shahrake Apadana': 5,'Shahr Ziba': 5,'West Ferdows Boulevard': 5, 'East Ferdows Boulevard': 5,'Central Janatabad': 5, 'Northern Janatabad': 5, 'Southern Janatabad': 5,
    'Eram':5,'North Program Organization':5,'Southern Program Organization':5,'Ayatollah Kashani':5,'Bolvar Ferdos':5,'Jannat Abad':5,'Sattari (Ab ta Noor)':5,
    'کن':5,'شهرک آپادانا':5,

    # منطقه ۶
    'Yousef Abad': 6, 'Amirabad': 6, 'Keshavarz Boulevard': 6, 'Fatemi': 6,'Mirza Shirazi':6,'Argentina':6,'Motahari (Modarres-shariati)':6,
    'Karimkhan': 6,'Valiasr':6,'Amirabad':6,'Villa':6,'Yousef Abad':6,'Gandhi':6,'Valiasr (Hemmat - Enghelab)':6,'Amir Abad':6,'Karim Khan':6,
    'جمالزاده':6,'گاندی':6,'توانیر':6,'Motahari (Valiasr)':6,'میرزای شیرازی':6,'سپهبد قرنی':6,'طالقانی':6,'قائم مقام فراهانی':6,'کردستان':6,
    'میدان آرژانتین':6,'انقلاب(از چهارراه ولیعصر تا میدان انقلاب)':6,'انقلاب(از Piche Shemran تا چهارراه ولیعصر)':6,'اسکندری شمالی':6,
    'چهارراه کالج':6,'Keshavarz':6,'وزراء':6,

    # منطقه ۷
    'Abbasabad': 7, 'Nezam Abad': 7, 'Heshmatieh': 7,'Bahar':7,'Haft Tir':7,'Northern Suhrawardi':7,'Garden of Saba':7,'Khaje Nasir':7,'Dabestan':7,'Namjoo Gorgan':7,
    'Sohrevardi Shomali':7,'Moallem':7,'Khaje Nezam':7,'Imam Hossein':7,'Sohrevardi J':7,'انقلاب(از Piche Shemran تا Imam Hossein)':7,
    'عباس آباد':7,'هفت تیر':7,'مفتح(از هفت تیر تا انقلاب)':7,'مفتح(از بهشتی تا هفت تیر)':7,'پل چوبی':7,
    # 9
    'Ostad Moein' : 9,'Si Metri Ji':9,
    'نیروی دریایی':9,'آپادانا':9,'استاد معین':9,'مهر آباد':9,'میدان فتح':9,'دستغیب':9,'شمشیری':9,'پدرثانی':9,
    # منطقه ۸
    'Narmak': 8, 'Majidieh': 8, 'Tehran Now': 8, 'Sabalan': 8,'Vahidieh':8,'Shahrake Madaen':8,'Taslihat':8,
    'Vahidiyeh':8,'Golbarg':8 ,'Majidieh Jonoobi':8,'دماوند(از Imam Hossein تا Vahidieh)':8,

    # منطقه ۱۰
    'Beryanak': 10,'Salsabil':10,'Qasr-od-Dasht':10,'Jeyhoon':10,'Hashemi':10,'Karoon':10,'Komeil':10,
    'Nawab':10,'هاشمی(از نواب تا یادگار)':10,'رودکی(سلسبیل)':10,'مالک اشتر':10,'Azadi (Ta navab)':10,
    'دامپزشکی(از یادگار تا آیت الله سعیدی)':10,'خوش' :10,'Ayatollah (Navab-Yadegar)':10,'آزادی(از میدان انقلاب تا نواب)':10,'Jeihoon':10,
    'کمیل':10,'زنجان':10,'کارون':10,'بریانک':10,'قصرالدشت':10, 'عارف':10,'نواب':10, 'دامپزشکی(از نواب تا یادگار)':10,
    'هاشمی(از یادگار تا آیت الله سعیدی)':10,

    # منطقه ۱۱
    'Enghelab': 11, 'Republic': 11, 'Amirieh': 11,'Moniriyeh' :11,'Northren Jamalzadeh':11,'Azarbaijan':11,'Amir Bahador':11,
    'Razi':11,'Eskandari':11,'Rah ahan':11,
    'حافظ':11,'قزوین(از نواب تا سه راه آذری)':11,'کارگر جنوبی':11,'جمهوری(از حافظ تا میدان جمهوری)':11,'وحدت اسلامی':11,
    'سی متری جی':11,'میدان حر' :11,'امام خمینی(از حسن آباد تا نواب)':11,'چهارراه لشگر':11,'اسکندری جنوبی':11,'امیریه':11,
    'گمرک':11,'منیریه':11,'مولوی(از وحدت اسلامی تا میدان رازی)':11,'پاستور':11,'ولیعصر(از چهارراه تا راه آهن)' :11,'هلال احمر':11,
    'خیابان قزوین(از ولیعصر تا نواب)':11,'حسن آباد':11,'میدان قزوین':11,'ابوسعید':11,'امین الملک':11,

    # منطقه ۱۲
    'Waterfall':12,'Hassan Abad':12,'میدان خراسان':12,'آذری':12,'مجاهدین اسلام':12,'مصطفی خمینی':12,'Molavi (Ghiam)':12,
    'خیام':12,'فردوسی':12,'بهارستان':12,'پانزده خرداد':12,'ایران' :12,'بازار':12,'شوش':12,'سعدی':12,'میدان فردوسی':12,
    'میدان قیام':12,'سنگلچ':12,'جمهوری(از بهارستان تا حافظ)':12,'لاله زار':12,'پارک شهر':12,'امیرکبیر':12,'منوچهری':12,
    'سه راه امین حضور':12,'ناصرخسرو':12,'باب همایون':12,

    # منطقه ۱۳
    'Piroozi': 13,'Air force':13,'Thirteen November':13,'Tehran No':13,'Niro Havaii':13,'آذربایجان':13,
    'Hefdah Shahrivar (Shohada)':13,'قصر فیروزه':13,
    # 14
    'Parastar':14,'Ahang':14,'Mahallati (Ahang)':14,'دهم فروردین جنوبی':14,'بلوار ابوذر جنوبی':14,'سلیمانیه':14,
    'پاسدار گمنام':14,'نبرد جنوبی':14,'دولاب':14,'بی سیم':14,

    # منطقه ۱۵
    'Afsarieh': 15, 'Khavaran': 15,'Atabak':15,'افسریه':15,'خاوران':15,'شهرک کاروان':15,'کیانشهر':15,'مشیریه':15,
    'سه راه افسریه':15,'مسعودیه':15,'سه راه ورامین':15,'بزرگراه بعثت(از فداییان تا سه راه افسریه)':15,'اتابک':15,
    'اصفهانک':15,'چیت سازی':15,

    # منطقه ۱۶
    'Naziabad': 16,'Railway':16,'Shoosh':16,'Aliabad South':16,'Javadiyeh':16,'Yakhchiabad':16,'Javadieh':16,
    'نازی آباد':16,'باغ خزانه':16,'خزانه':16,'علی آباد':16,'شهید رجایی(از شوش تا آزادگان)':16,'یاخچی آباد':16,
    'میدان بهمن':16,'بزرگراه بعثت(از میدان بهمن تا فداییان اسلام)':16,'بلورسازی':16,'تندگویان':16,
    # 17
    'Qazvin Imamzadeh Hassan':17,'Fallah':17,'Azari':17,'Boloorsazi':17, 'فلاح':17,'زمزم':17,'امام زاده حسن':17,
    'برادران حسنی(قلعه مرغی)':17,'آیت الله سعیدی':17,
    # 18
    'Yaftabad':18,'Shadabad':18,'بلوار خلیج فارس':18,'شاد آباد':18,'یافت آباد':18,
    # 19
    'Baghestan':19,'Salehabad':19, 'خانی آباد نو':19,'عبدل آباد':19,'نعمت آباد':19,'جاده قم':19,'قلعه مرغی(بوستان ولایت)':19,
    'شهید کاظمی':19,
    # منطقه ۲۰
    'Ray': 20, 'Ray - Montazeri': 20, 'Ray - Pilgosh': 20,'پل سیمان':20,'امین آباد':20,'ویلا شهر':20,'Dolat آباد':20,'ری':20,
    'فداییان اسلام(از آزادگان تا میدان شهر ری)':20,'سعید آباد':20,'جوانمرد قصاب':20,'چشمه علی':20,'ابن بابویه':20,'تقی آباد':20,
    # 21
    'Tehransar':21,'Shahrake Azadi':21,'تهرانسر':21,'وردآورد':21,'آزادگان(از آیت الله سعیدی تا جاده قدیم کرج)':21,
    # منطقه ۲۲
    'Northern Chitgar': 22, 'Southern Chitgar': 22, 'Dehkade Olampic': 22,'Shahrake Shahid Bagheri':22,'Persian Gulf Martyrs Lake':22
    ,'Golestan':22,'Southern Chitgar':22,'Kook':22,'Azadshahr':22,'Zibadasht':22,'چیتگر':22,
}
df['district_number']=df['Neighborhood (English)'].map(neighborhood_to_district)
df['district_number'] = df['district_number'].fillna(0)
suburb ={
  'سایر',
  'شهر ری',
 'جاده احمدآباد مستوفی'
}
pattern = '|'.join(suburb)
df['is_suburb']=df['Neighborhood (English)'].str.contains(pattern, case=False , na=False)

#? Based on the tests I conducted, I observed this pattern in Districts 1, 2, and 3.
#? However, Districts 2 and 3 introduced significant noise and reduced the model's accuracy.
#? Therefore, I defined the condition for District 1 and properties with an area greater than 50 square meters.
df['top_area']= False
msk =(
    (df['Base Area'] > 50) &
    (df['district_number']==1)
)
df.loc[msk , 'top_area'] = True     

#? To prepare the text-based Address feature for the neural network,
#? I applied one-hot encoding using get_dummies.
ch_cols = ['Neighborhood (English)']
df = pd.get_dummies(df , columns=ch_cols)


# print("Original:", df['Total Price'].skew())
# print("Log:", np.log1p(df['Total Price']).skew())

x =df.drop('Total Price',axis=1)
y = np.log1p(df['Total Price'])

train_x , temp_x , train_y , temp_y = train_test_split(x , y , random_state=42 , test_size=0.4)
x_val , test_x , y_val , test_y = train_test_split(temp_x,temp_y , random_state=42 , test_size=0.5)

#? Using target encoding, I calculated the average price for each district
#? in a way that prevents data leakage.
#? I used Smoothed Target Encoding to reduce the effect of districts
#? with a small number of samples.


def target_encoding(x_train , x_val ,x_test, y_train ,m = 50):

    train_df = x_train.copy()
    train_df['target']= y_train

    global_mean = train_df['target'].mean()

    stats = train_df.groupby('district_number')['target'].agg(['count' , 'mean'])

    stats["smooth"] = (
            stats["count"]*stats["mean"] + m*global_mean
        )/(stats["count"]+m)
    
    x_train = x_train.copy()
    x_val = x_val.copy()
    x_test = x_test.copy()

    x_train['district_target_mean'] = (x_train['district_number'].map(stats['smooth']))
    x_val['district_target_mean'] = (x_val['district_number'].map(stats['smooth']))
    x_test['district_target_mean'] = (x_test['district_number'].map(stats['smooth']))

    x_train['district_target_mean'] = (x_train['district_target_mean'].fillna(global_mean))
    x_val['district_target_mean'] = (x_val['district_target_mean'].fillna(global_mean))
    x_test['district_target_mean'] = (x_test['district_target_mean'].fillna(global_mean))
    return x_train , x_val,x_test

train_x, x_val, test_x = target_encoding(
    train_x,
    x_val,
    test_x,
    train_y
)


scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
train_x = scaler_x.fit_transform(train_x)
train_y =scaler_y.fit_transform(train_y.values.reshape(-1,1))
test_x = scaler_x.transform(test_x)
test_y = scaler_y.transform(test_y.values.reshape(-1,1))
x_val = scaler_x.transform(x_val)
y_val = scaler_y.transform(y_val.values.reshape(-1,1))

train_x = torch.tensor(train_x,dtype=torch.float32)
train_y = torch.tensor(train_y,dtype=torch.float32)
x_val = torch.tensor(x_val,dtype=torch.float32)
y_val = torch.tensor(y_val,dtype=torch.float32)
test_x = torch.tensor(test_x,dtype=torch.float32)
test_y = torch.tensor(test_y,dtype=torch.float32)

epochs = 500
batch_size = 128
lr = 0.001
loss_function = nn.L1Loss()


class nueralnetwork(nn.Module):
    def __init__(self , input_dim = train_x.shape[1], output_dim = train_y.shape[1]):
        super().__init__()
        torch.manual_seed(2026)
        self.fc1 = nn.Linear(input_dim , 52)
        self.fc2 = nn.Linear(52,94)
        self.fc3 = nn.Linear(94,258)
        self.fc4 = nn.Linear(258 , output_dim)

        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        
        return x
    
model = nueralnetwork()

optimizer = optim.Adam(model.parameters(),lr=lr)
train_data = TensorDataset(train_x,train_y)
train_loader = DataLoader(train_data, batch_size=batch_size , shuffle=True)
def train_network(model , loss_function , optimizer , epochs , train_loader):
    train_loss_list =[]
    total_loss=[]
    best_val_loss = float('inf')
    counter =0
    patience =50
    best_model_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batch = 0

        for batch_x , batch_y in train_loader:
            output_data = model(batch_x)
            loss = loss_function(output_data , batch_y)
            train_loss_list.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            num_batch+=1
        epoch_loss = train_loss/num_batch
        total_loss.append(epoch_loss)

        model.eval()
        with torch.inference_mode():
            output_val = model(x_val)
            loss_val = loss_function(output_val,y_val)
        if loss_val.item() < best_val_loss :
            best_val_loss = loss_val.item()
            counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            counter+=1
            if counter>= patience :
                print(f"Early stopping at epoch {epoch + 1}")
                break
        if (epoch + 1) % 25 == 0:
                            
                            print(
                                f"Epoch: {epoch + 1} | "
                                f"Train Loss: {epoch_loss:.4f} | "
                                f"Val Loss: {loss_val.item():.4f}"
                            )
    model.load_state_dict(best_model_state)
    return total_loss




def eval_model(model, train_x, train_y, test_x, test_y, scaler_y):

    model.eval()

    with torch.inference_mode():

        # Predictions
        train_pred = model(train_x)
        test_pred = model(test_x)

    
    train_pred = train_pred.numpy()
    test_pred = test_pred.numpy()

    train_y = train_y.numpy()
    test_y = test_y.numpy()

    
    train_pred = scaler_y.inverse_transform(train_pred)
    test_pred = scaler_y.inverse_transform(test_pred)

    train_y = scaler_y.inverse_transform(train_y)
    test_y = scaler_y.inverse_transform(test_y)

    
    train_pred = np.expm1(train_pred)
    test_pred = np.expm1(test_pred)

    train_y = np.expm1(train_y)
    test_y = np.expm1(test_y)

    # -------------------------
    # Training Metrics
    # -------------------------

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
    scaler_y
)

#? Shared features used for Transfer Learning
transfer_x = df[['Base Area','district_number','is_suburb','top_area']]


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