import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. Verileri Yükleme ve İşleme ---
# Veriyi DataFrame'e aktar
df = pd.read_csv('data.csv')
columns = ["Tarih", "TP_DK_USD_S_YTL", "TP_DK_USD_A_YTL", "TP_DK_EUR_S_YTL", "TP_DK_EUR_A_YTL", 
           "TP_DK_GBP_S_YTL", "TP_DK_GBP_A_YTL", "TP_DK_CHF_S_YTL", "TP_DK_CHF_A_YTL", 
           "TP_DK_JPY_S_YTL", "TP_DK_JPY_A_YTL"]
df.columns = columns

# Gerekli sütunları seç (sadece satış fiyatları)
df = df[["TP_DK_USD_S_YTL", "TP_DK_EUR_S_YTL", "TP_DK_GBP_S_YTL", "TP_DK_CHF_S_YTL", "TP_DK_JPY_S_YTL"]]

# Veriyi ölçeklendir
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Eğitim ve test verilerini ayır
train_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

# Eğitim verilerini oluştur
X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])  # Son 60 gün
    y_train.append(train_data[i, 0])   # USD/TRY'yi hedef olarak seç

X_train, y_train = np.array(X_train), np.array(y_train)

# Test verilerini oluştur
X_test = []
y_test = []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# --- 2. LSTM Modelini Oluşturma ---
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Modeli derle
model.compile(optimizer='adam', loss='mean_squared_error')

# --- 3. Modeli Eğitme ---
model.fit(X_train, y_train, batch_size=8, epochs=50)

# --- 4. Test Verileri ile Tahmin Yapma ---
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 4))), axis=1))[:, 0]

# Gerçek USD/TRY değerlerini geri döndür
real_values = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1))[:, 0]

# --- 5. Sonuçları Görselleştirme ---
plt.figure(figsize=(16, 8))
plt.plot(real_values, color='blue', label='Gerçek USD/TRY')
plt.plot(predictions, color='red', label='Tahmin USD/TRY')
plt.title('USD/TRY Kuru Tahmini')
plt.xlabel('Gün')
plt.ylabel('USD/TRY')
plt.legend()
plt.show()

# --- 6. Modelin Başarısını Ölçme ---
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(real_values, predictions)
print(f"Ortalama Kare Hata (MSE): {mse:.4f}")

# --- 7. Görseli Kaydetme ---
plt.figure(figsize=(16, 8))
plt.plot(real_values, color='blue', label='Gerçek USD/TRY')
plt.plot(predictions, color='red', label='Tahmin USD/TRY')
plt.title('USD/TRY Kuru Tahmini')
plt.xlabel('Gün')
plt.ylabel('USD/TRY')
plt.legend()
plt.savefig('usd_try.png')

# --- 8. Geleceği Tahmin Etme ---
last_60_days = test_data[-60:]  # Test verilerinin son 60 günü
last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], last_60_days.shape[1]))

future_prediction = model.predict(last_60_days)
future_prediction = scaler.inverse_transform(np.concatenate((future_prediction, np.zeros((future_prediction.shape[0], 4))), axis=1))[:, 0]

print(f"Gelecekte tahmin edilen USD/TRY kuru: {future_prediction[0]:.4f}")
