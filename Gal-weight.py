# Tahap Persiapan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

z = 0.07894
teta = 1.7 /z
rapusat = 29.107189999999996
decpusat = 1.0562099999999965

Ramax = rapusat + (teta/60)
Ramin = rapusat - (teta/60)
Decmax = decpusat + (teta/60)
Decmin = decpusat - (teta/60)

print(Ramax, Ramin, Decmax, Decmin)

df = pd.read_csv('/content/DataAwalAgustus2024_bismaridho.csv')

df = df.rename(columns={'Column1': 'zspek'})
df = df.rename(columns={'z': 'zphoto'})
df = df.rename(columns={'Column2': 'z'})

df

#Filter data

filtered_data = df[(df['zphoto'] >= 0) & (df['zspek'] >= 0) & (df['zspek'] < 1) ]
filtered_data = filtered_data.reset_index(drop=True)

filtered_data.describe()

filtered_data1 = df[(df['zphoto'] >= 0) & (df['zspek'] >= 0.074) & (df['zspek'] < 0.5) ]
filtered_data1 = filtered_data1.reset_index(drop=True)

filtered_data1.describe()

##Aplikasi Metode 1-D AKM

#Jika menggunakan Aturan Silverman

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Fungsi Gaussian kernel
def gaus(t):
    return np.exp(-(t**2)/2) / np.sqrt(2 * np.pi)

# Perhitungan bandwidth standar dengan aturan Silverman
band = 1.06 * df1['zspek'].std() * (len(df1)**(-1/5))

# Aturan Scott untuk bandwidth
#band = 3.49 * df1['zspek'].std() * (len(df1)**(-1/3))

# Pilot Density Estimation (f00)
def f00(x1, band):
    B = 0
    for j in range(len(df1)):
        X = (x1 - df1.at[j, 'zspek']) / band
        gauss = gaus(X)
        B += gauss
    return B / (len(df1) * band)

# Perhitungan log-likelihood untuk gamma
def calculate_gamma(df1, band):
    log_sum = 0
    for i in range(len(df1)):
        B = 0
        for j in range(len(df1)):
            X = (df1.at[i, 'zspek'] - df1.at[j, 'zspek']) / band
            gauss = gaus(X)
            B += gauss
        hasil = B / (len(df1) * band)
        log_sum += np.log10(hasil)
    return np.power(10, log_sum / len(df1))

# Hitung gamma sekali untuk keseluruhan dataset
gamma = calculate_gamma(df1, band)

# Fungsi AKM dengan bandwidth adaptif
def adaptive_kernel_method(x2, df1, band, gamma):
    C = 0
    for k in range(len(df1)):
        f00_k = f00(df1.at[k, 'zspek'], band)
        adaptive_bandwidth = band * np.sqrt(gamma / f00_k)
        W = (x2 - df1.at[k, 'zspek']) / adaptive_bandwidth
        gauss = gaus(W) / adaptive_bandwidth
        C += gauss
    return C / len(df1)

# Membuat range nilai z untuk diprediksi
test = np.linspace(0.074, 0.086, 200)

# Hitung prediksi z menggunakan fungsi adaptive kernel method
zpred = np.array([adaptive_kernel_method(t, df1, band, gamma) for t in test])

# Plot histogram dengan Matplotlib dan hasil prediksi dengan adaptive kernel method
plt.subplots(figsize=(12, 6))

plt.hist(df1['zspek'], bins=26, alpha=0.8,
         histtype='bar', color='blue',
         edgecolor='black', density=True)
plt.plot(test, zpred, color='red')
plt.xlim(0.074, 0.086)
plt.xlabel('z')
plt.ylabel('Density')
plt.title('Histogram of redshift with 1D AKM with Silverman Method')
plt.show()

# Plot menggunakan Plotly
a = pd.DataFrame(dict(
    z=test,
    n=zpred
))

fig = px.line(a, x='z', y='n', title='Adaptive Kernel Density Estimation with Plotly')
fig.show()


#Jika menggunakan Aturan Sheather-Jones

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Fungsi Gaussian kernel
def gaus(t):
    return np.exp(-(t**2)/2) / np.sqrt(2 * np.pi)

# Gunakan statsmodels untuk menghitung bandwidth menggunakan Sheather-Jones
kde = sm.nonparametric.KDEUnivariate(df1['zspek'])
kde.fit(kernel='gau', bw='scott')  # Dapat diganti dengan 'scott', 'silverman', atau 'normal_reference'

# Bandwidth Sheather-Jones yang dihitung oleh statsmodels
stjj_sj = kde.bw

# Pilot Density Estimation (f00) menggunakan stjj_sj
def f00_sj(x1, stjj_sj):
    B = 0
    for j in range(len(df1)):
        X = (x1 - df1.at[j, 'zspek']) / stjj_sj
        gauss = gaus(X)
        B += gauss
    return B / (len(df1) * stjj_sj)

# Perhitungan log-likelihood untuk gamma menggunakan stjj_sj
def calculate_gamma_sj(df1, stjj_sj):
    log_sum = 0
    for i in range(len(df1)):
        B = 0
        for j in range(len(df1)):
            X = (df1.at[i, 'zspek'] - df1.at[j, 'zspek']) / stjj_sj
            gauss = gaus(X)
            B += gauss
        hasil = B / (len(df1) * stjj_sj)
        log_sum += np.log10(hasil)
    return np.power(10, log_sum / len(df1))

# Hitung gamma sekali untuk keseluruhan dataset menggunakan stjj_sj
gamma_sj = calculate_gamma_sj(df1, stjj_sj)

# Fungsi AKM dengan bandwidth adaptif menggunakan stjj_sj
def adaptive_kernel_method_sj(x2, df1, stjj_sj, gamma_sj):
    C = 0
    for k in range(len(df1)):
        f00_k = f00_sj(df1.at[k, 'zspek'], stjj_sj)
        adaptive_bandwidth = stjj_sj * np.sqrt(gamma_sj / f00_k)
        W = (x2 - df1.at[k, 'zspek']) / adaptive_bandwidth
        gauss = gaus(W) / adaptive_bandwidth
        C += gauss
    return C / len(df1)

# Membuat range nilai z untuk diprediksi
test = np.linspace(0.074, 0.086, 150)

# Hitung prediksi z menggunakan fungsi adaptive kernel method dengan Sheather-Jones bandwidth
zpred_sj = np.array([adaptive_kernel_method_sj(t, df1, stjj_sj, gamma_sj) for t in test])

# Plot histogram dengan Matplotlib dan hasil prediksi dengan adaptive kernel method Sheather-Jones
plt.subplots(figsize=(12, 6))

plt.hist(df1['zspek'], bins=26, alpha=0.8,
         histtype='bar', color='blue',
         edgecolor='black', density=True)
plt.plot(test, zpred_sj, color='red')
plt.xlim(0.074, 0.086)
plt.xlabel('z')
plt.ylabel('Density')
plt.title('Histogram Redshift dengan 1D AKM')
plt.show()

# Plot menggunakan Plotly
a_sj = pd.DataFrame(dict(
    z=test,
    n=zpred_sj
))

fig_sj = px.line(a_sj, x='z', y='n', title='Adaptive Kernel Density Estimation (Sheather-Jones) with Plotly')
fig_sj.show()


#Peta awal

rap =  29.07064  #29.107189999999996
decp =  1.05082 #1.0562099999999965
#29.0706414923225	, df1['dec'], 1.0508166673879)

plt.scatter(filtered_data['ra'],filtered_data['dec'], label ='Spektro',marker='.',color='black',alpha=0.5)
plt.xlabel("RA")
plt.ylabel("Dec")
plt.title("Peta persebaran Keseluruhan Data")

Spektro = filtered_data[(filtered_data['ra']==29.0706414923225) & (filtered_data['dec']==1.0508166673879)]
plt.scatter(filtered_data['ra'],filtered_data['dec'], label = 'spektro', marker='.', color='black',alpha=0.5)
plt.scatter(Spektro['ra'],Spektro['dec'],marker = '.', color='red')
plt.title("Titik Pusat Gugus Galaksi Abell 0279")
plt.xlabel("RA")
plt.ylabel("Dec")
plt.show()

#Membatasi data

rap =  29.07064  #29.107189999999996
decp =  1.05082 #1.0562099999999965
#29.0706414923225	, df1['dec'], 1.0508166673879)

plt.scatter(filtered_data['ra'],filtered_data['dec'], label ='Spektro',marker='.',color='black',alpha=0.5)
plt.xlabel("RA")
plt.ylabel("Dec")
plt.title("Peta persebaran Keseluruhan Data")

Spektro = filtered_data[(filtered_data['ra']==29.0706414923225) & (filtered_data['dec']==1.0508166673879)]
plt.scatter(filtered_data['ra'],filtered_data['dec'], label = 'spektro', marker='.', color='black',alpha=0.5)
plt.scatter(Spektro['ra'],Spektro['dec'],marker = '.', color='red')
plt.title("Titik Pusat Gugus Galaksi Abell 0279")
plt.xlabel("RA")
plt.ylabel("Dec")
plt.show()

df1 = filtered_data[(filtered_data['zspek'] >= 0.070) & (filtered_data['zspek'] <= 0.086)]
df1 = df1.reset_index(drop=True)
df1

plt.figure(figsize=(12,6))
plt.scatter(filtered_data['ra'],filtered_data['dec'],color='black',alpha=0.5)
plt.scatter(df1['ra'],df1['dec'], label='A0279',color='red',alpha=1)
plt.scatter(Spektro['ra'],Spektro['dec'], color='yellow')
plt.title("Distribusi Ruang Calon Gugus Galaksi Abell 0279")
plt.xlabel("RA")
plt.ylabel("Dec")
plt.show()
