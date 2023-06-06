#!/usr/bin/env python
# coding: utf-8

# In[15]:


#import dulu library yang akan digunakan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


#membaca data csv
df = pd.read_csv("Data-untuk_Naive-Bayes-Classifier.csv")
df.head() #menampilkan data default 5 rows


# In[17]:


#mengindeks data yang akan digunakan dengan iloc, ambil semua data baris dan kolom (:), kecuali kolom terakhir (:-1). kemudian konversihasil dalam bentuk array dengan (.values)
X = df.iloc[:, :-1].values

#ambil semua baris (:) dan hanya kolom ke-2 (diabetes) dari DataFrame 
y = df.iloc[:, 2].values


# ## Modeling

# In[18]:


#split data menjadi data training dan data testing
#X = fitur pemrediksi variabel target (y)
#X_train dan y_train = dataset latih yang berisi sebagian data asli yang akan digunakan untuk melatih model.
#X_test dan y_test akan menjadi dataset uji
#test_size=0.33 menunjukkan bahwa data uji akan memiliki ukuran sebesar 33% (atau sekitar sepertiga) dari keseluruhan dataset
#random_state=42 digunakan untuk memberikan nilai tetap (42 dalam kasus ini) ke generator angka acak

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[8]:


model = GaussianNB() # Model ini cocok untuk digunakan ketika fitur-fitur input (X) dianggap mengikuti distribusi Gaussian (normal).

#fit() digunakan untuk melatih (mengasah) model pada data latih
#X_train dan y_train adalah dataset latih yang digunakan untuk melatih model data baru
model.fit(X_train, y_train)


# In[19]:


y_pred = model.predict(X_test)
#X_test adalah dataset uji yang tidak terlihat oleh model selama pelatihan.
#Metode predict() akan menghasilkan prediksi nilai target berdasarkan fitur-fitur dari X_test.


# In[20]:


#menampilkan hasil dari y_pred
y_pred


# In[22]:


accuracy = accuracy_score(y_test, y_pred)*100
#Parameter pertama (y_test) adalah nilai target yang sebenarnya dari dataset uji, 
#dan parameter kedua (y_pred) adalah nilai target yang diprediksi oleh model.
#hasil akurasi tersebut dikalikan dengan 100 untuk mengonversinya menjadi persentase.

accuracy #menampilkan hasil uji


# In[23]:


from sklearn.metrics import classification_report, confusion_matrix
#classification_report digunakan untuk menghasilkan laporan klasifikasi
#confusion_matrix digunakan untuk menghasilkan matriks kebingungan yang menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas.

print(classification_report(y_test, y_pred))
#Parameter pertama (y_test) adalah nilai target yang sebenarnya dari dataset uji, 
#dan parameter kedua (y_pred) adalah nilai target yang diprediksi oleh model.


# In[13]:


print(confusion_matrix(y_test, y_pred))


# In[25]:


#Visualisasi Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Mendapatkan matriks kebingungan
cm = confusion_matrix(y_test, y_pred)

# Membuat heatmap dari matriks kebingungan
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True,fmt='d', cmap="Blues")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# ## Prediction

# In[26]:


#0 untuk hasil negatif
#1 untuk hasil positif
print(model.predict([[45, 100]]))

