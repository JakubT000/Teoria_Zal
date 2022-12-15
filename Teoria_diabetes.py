#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importowanie potrzebnych bibliotek
#pip install missingno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Załadowanie danych
df = pd.read_csv("C:/Users/siepa/OneDrive/Pulpit/diabetes.csv")
df.head(3)


# In[3]:


#Szukanie brakujących wartości
df.info()


# In[4]:


df.describe()


# In[5]:


#Przygotowanie braków danych do dalszej analizy
df.loc[df["Glucose"] == 0.0, "Glucose"] = np.NAN
df.loc[df["BloodPressure"] == 0.0, "BloodPressure"] = np.NAN
df.loc[df["SkinThickness"] == 0.0, "SkinThickness"] = np.NAN
df.loc[df["Insulin"] == 0.0, "Insulin"] = np.NAN
df.loc[df["BMI"] == 0.0, "BMI"] = np.NAN

df.isnull().sum()[1:6]


# In[7]:


#Barplot kompletności danych
mno.bar(df,color="pink")


# In[9]:


#Macierz braków danych
mno.matrix(df, figsize = (20, 10))


# In[10]:


#Korelacja braków danych
mno.heatmap(df)


# In[11]:


#Imputacja losowa
kolumny_z_brakami = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
def imputacja_losowa(df, kolumna):

    liczba_brakow = df[kolumna].isnull().sum()
    wartosci_niezerowe = df.loc[df[kolumna].notnull(), kolumna]
    df.loc[df[kolumna].isnull(), kolumna + '_imp'] = np.random.choice(wartosci_niezerowe, liczba_brakow, replace = True)
    
    return df

for kolumna in kolumny_z_brakami:
    df[kolumna + '_imp'] = df[kolumna]
    df = imputacja_losowa(df, kolumna)
df    


# In[12]:


#Regresja deterministyczna
#Tworzę ramke danych jedynie z imputowanymi kolumnami
DR_data = pd.DataFrame(columns = ["DR" + nazwa for nazwa in kolumny_z_brakami])

for kolumna in kolumny_z_brakami:
        
    DR_data["DR" + kolumna] = df[kolumna + "_imp"]
    parameters = list(set(df.columns) - set(kolumny_z_brakami) - {kolumna + '_imp'})
    
    #Tworzę model do estymacji brakujących wartości
    model = linear_model.LinearRegression()
    model.fit(X = df[parameters], y = df[kolumna + '_imp'])
    
    #Uzupełniam drugą tabelę
    DR_data.loc[df[kolumna].isnull(), "DR" + kolumna] = model.predict(df[parameters])[df[kolumna].isnull()]


# In[13]:


DR_data


# In[14]:


#Mamy brak braków danych
mno.matrix(DR_data, figsize = (20,5))


# In[15]:


#Wizualizacja dla regresji deterministycznej
sns.set(font_scale = 1.5)
fig, axes = plt.subplots(nrows = 2, ncols = 2)
fig.set_size_inches(10, 10)

for index, zmienna in enumerate(["Insulin", "SkinThickness"]):
    sns.distplot(df[zmienna].dropna(), kde = False, ax = axes[index, 0])
    sns.distplot(DR_data["DR" + zmienna], kde = False, ax = axes[index, 0], color = 'red')
    axes[index, 0].set(xlabel = zmienna+ " / " + zmienna + '_impDR')
    
    
    sns.boxplot(data = pd.concat([df[zmienna],DR_data["DR" + zmienna]], axis = 1),
                ax = axes[index, 1])
    
    
    plt.tight_layout()


# In[16]:


#Regresja stochastyczna
#Tworzę ramke danych jedynie z imputowanymi kolumnami
SR_data = pd.DataFrame(columns = ["SR" + nazwa for nazwa in kolumny_z_brakami])

for kolumna in kolumny_z_brakami:
        
    SR_data["SR" + kolumna] = df[kolumna + '_imp']
    parameters = list(set(df.columns) - set(kolumny_z_brakami) - {kolumna + '_imp'})
    #Tworzę bazowy model
    model = linear_model.LinearRegression()
    model.fit(X = df[parameters], y = df[kolumna + '_imp'])
    
    #Dodaję zakłócenie
    predict = model.predict(df[parameters])
    std_error = (predict[df[kolumna].notnull()] - df.loc[df[kolumna].notnull(), kolumna + '_imp']).std()
    
    
    random_predict = np.random.normal(size = df[kolumna].shape[0], 
                                      loc = predict, 
                                      scale = std_error)
    SR_data.loc[(df[kolumna].isnull()) & (random_predict > 0), "SR" + kolumna] = random_predict[(df[kolumna].isnull()) & 
                                                                            (random_predict > 0)]


# In[17]:


SR_data


# In[18]:


#Wizualizacja dla regresji stochastycznej
sns.set(font_scale = 1.5)
fig, axes = plt.subplots(nrows = 2, ncols = 2)
fig.set_size_inches(10, 10)

for index, zmienna in enumerate(["Insulin", "SkinThickness"]):
    sns.distplot(df[zmienna].dropna(), kde = False, ax = axes[index, 0])
    sns.distplot(SR_data["SR" + zmienna], kde = False, ax = axes[index, 0], color = 'red')
    axes[index, 0].set(xlabel = zmienna+ " / " + zmienna + '_impSR')
    
    
    sns.boxplot(data = pd.concat([df[zmienna],SR_data["SR" + zmienna]], axis = 1),
                ax = axes[index, 1])
    
    
    plt.tight_layout()


# In[19]:


#Wykresy pudełkowe dla SR i DR
sns.set(font_scale = 2)
fig, axes = plt.subplots(2)
fig.set_size_inches(10,20)
for index, zmienna in enumerate(["Insulin", "SkinThickness"]):
    
    
    sns.boxplot(data = pd.concat([df[zmienna],DR_data["DR" + zmienna],SR_data["SR" + zmienna]], axis = 1),
                ax = axes[index])
    
plt.show()


# In[21]:


#Statystki podsumowujące
pd.concat([df[["Insulin", "SkinThickness"]], SR_data[["SRInsulin", "SRSkinThickness"]],DR_data[["DRInsulin", "DRSkinThickness"]]], axis = 1).describe().T


# In[ ]:




