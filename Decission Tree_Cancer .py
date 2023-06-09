#!/usr/bin/env python
# coding: utf-8

# In[15]:


#library yang digunakan
import numpy as np # untuk perhitungan aljabar linear
import pandas as pd # untuk data prosesing
import matplotlib.pyplot as plt #untuk visualisasi data


# In[16]:


#load & read dataset
data = pd.read_csv('dataset_cancer.csv')
data.head()


# In[17]:


#gali informasi data
data.info()


# In[18]:


#gali informasi statistik data
data.describe()


# In[19]:


#cek dimensi dataset
data.shape


# In[20]:


#periksa missing value pada data
MV = data.isnull().sum()
MV


# In[21]:


#drop kolom yang tidak terpakai/ berpengaruh 
data = data.drop(["id"], axis = 1)
data = data.drop(["Unnamed: 32"], axis = 1)
data.head()


# In[22]:


#diagnosis M (Malignant = tumor ganas)
mgn = data[data.diagnosis == "M"] 
mgn.head()


# In[25]:


#diagnosis B (Benign = tumor jinak)
bng = data[data.diagnosis == "B"]
bng.head()


# In[26]:


#visualisasikan data 
plt.title("Tumor Ganas vs Tumor jinak")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.scatter(mgn.radius_mean, mgn.texture_mean, color = 'red', label = "Tummor Ganas", alpha = 0.3)
plt.scatter(bng.radius_mean, bng.texture_mean, color = 'blue', label = "Tummor Jinak", alpha = 0.3)
plt.legend()
plt.show()


# 
# 
# ![image.png](attachment:image.png)
# 
# ##Arti Algoritma Pohon Keputusan
# 
# a. Model pohon keputusan di mana variabel target menggunakan sekumpulan nilai diskrit diklasifikasikan sebagai Pohon Klasifikasi.
# 
# b. Pada pohon-pohon ini, setiap simpul, atau daun, mewakili label kelas, sementara cabang-cabangnya mewakili gabungan fitur yang mengarah ke label kelas.
# 
# c. Pohon keputusan di mana variabel target mengambil nilai kontinu, biasanya berupa angka, disebut Pohon Regresi.
# 
# d. Kedua jenis ini biasanya disebut bersama sebagai CART (Classification and Regression Tree).

# In[28]:


#Decission Tree with Sklean
data.diagnosis = [1 if i == "M" else 0 for i in data.diagnosis]


# In[29]:


x = data.drop(["diagnosis"], axis = 1)
y = data.diagnosis.values


# In[30]:


# Normalisasi:
x = (x - np.min(x)) / (np.max(x) - np.min(x))


# In[32]:


#split data ke train dan test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[33]:


#import library sklearn
from sklearn.tree import DecisionTreeClassifier


# In[34]:


dt = DecisionTreeClassifier()


# In[35]:


dt.fit(x_train, y_train)


# In[38]:


# prediction
dt.score(x_test, y_test)


# In[ ]:




