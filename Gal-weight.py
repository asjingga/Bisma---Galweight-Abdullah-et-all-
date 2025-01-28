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

#Penggunaan metode 1-D AKM

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

#Perhitungan Jarak Lumino

import numpy as np
z = 0.0797182
dh = 300000/70.4
omegar = 0
omegam = 0.27
omegal = 0.73
omegak = 0
E = np.sqrt((omegam*((1+z)**(3)))+(omegal))

dc=dh*(z/E)
dl = (1+z)*dc
print("Jarak Hubble adalah sebesar" ,dh , "Mpc")
print("Jarak Comoving adalah sebesar" ,dc , "Mpc")
print("Jarak Luminositas adalah sebesar",dl, "Mpc")

#Jarak Proyeksi dan Kecepatan Peculiar

def jarak (ra,rap,dec,decp):
  delra=np.abs(ra-rap)
  cos = np.cos(np.radians(90-dec))*np.cos(np.radians(90-decp))
  sin = np.sin(np.radians(90-dec))*np.sin(np.radians(90-decp))*np.cos(np.radians(delra))
  C = sin + cos
  ar = np.arccos(C)
  return np.tan(ar)*dl

df1 = filtered_data[(filtered_data['zspek'] >= 0.070) & (filtered_data['zspek'] <= 0.086)]
df1 = df1.reset_index(drop=True)
df1.to_csv('hasil_dataawal.csv', index=False)

c = 300000
df1['vpec'] = (c*(df1['zspek']-z))/(1+z)
df1['jarakp']=jarak(df1['ra'], rap	, df1['dec'], decp)

#Menentukan plot distribusi ruang fase

plt.scatter(df1['jarakp'],df1['vpec'])
plt.ylabel('V_pec (km/s)')
plt.xlabel('Jarak Proyeksi (Mpc)')
plt.title ('Distribusi Ruang Fase')
plt.ylim(-2000,2000)

#Menentukan plot distrubusi bobot dinamikal (dalam fungsi R_proj serta V_pec)
def kerapatan (jumlah, r1, r2):
  luas1=np.pi*(r1**2)
  luas2=np.pi*(r2**2)
  total=luas2-luas1
  return jumlah/total

rapat2 = []
jarak2 = []

for i in range (25, 575, 25):
  jum2 = df1[(df1['jarakp'] <= i/100)&(df1['jarakp'] > (i - 25)/100)]
  print(len(jum2))
  jarak2.append(i/100)
  rapat2.append(kerapatan(len(jum2), (i-25)/100, i/100))

df1['vpec^2']=(df1['vpec']**2)
df1['abv']=np.abs(df1['vpec'])

disp=[]
R=[]

for i in range (25, 575, 25):
  dis1 = df1[(df1['jarakp'] <= i/100 )& (df1['jarakp'] > (i-25)/100)]
  sig = len(dis1)
  print(sig)
  disp.append(np.sqrt(dis1['vpec^2'].mean()))
  R.append(i/100)

disp3 = []

for i in range (50, 570, 25):
  dis = df1[(df1['jarakp'] <= i/100) & (df1['jarakp'] > (i-25)/100)]
  sigma = len(dis)
  print(sigma)
  disp3.append(np.sqrt(dis['vpec^2'].mean()))

s=0
for i in range (len(disp3)):
  s = s + disp3[i]
  print(s)
n= s/len(disp3)
print("n:", n)

nu = (disp[0]/n) - 1
nu

D = []
for j in range (len(jarak2)):
  atas = rapat2[j]*disp[j]
  bawah = np.power(jarak2[j], nu)
  total = atas/bawah
  D.append(total)

D = [abs(x) for x in D]

len(D)

plt.scatter(jarak2, D)

Dnorm = 0
h = 0.25  # Lebar interval

for k in range(0, len(D)-1):
    tambah = (D[k] + D[k+1]) * h / 2
    print(tambah)
    Dnorm = Dnorm + tambah

print("Perkiraan integral menggunakan aturan Trapezoidal:", Dnorm)

norm = D/Dnorm
norm

print(jarak2)

#Simpan file untuk analisis matlab
from scipy.io import savemat
datajarak = {
    'jarak': np.array(jarak2),    # Konversi ke array numpy agar kompatibel
    'NR': np.array(norm)   # Konversi ke array numpy
}

savemat('datajarak.mat', datajarak)

vz = []
Nvz = []

for i in range (100,1501,100):
  j = df1[(df1['abv'] <= i)&(df1['abv'] > i-100)]
  print(len(j))
  vz.append(i)
  Nvz.append(len(j))


df1['jarakp']

print(vz)

Nvz

plt.scatter(vz, Nvz)

Nnorm = []
for k in range (len(Nvz)):
  Nnorm.append(Nvz[k]/len(df1))

print(Nnorm)


#Simpan file untuk analisis matlab
datavpec = {
    'vz': np.array(vz),    # Konversi ke array numpy agar kompatibel
    'Nvz': np.array(Nnorm)   # Konversi ke array numpy
}

savemat('datavpec.mat', datavpec)

plt.scatter(vz, Nnorm)
plt.ylabel('D/N')
plt.xlabel('v (km/s)')

#Fitting data berdasarkan bobot dinamikal
result_fit2 = fit2(df1['abv'])
result_fit1 = fit1(df1['jarakp'])

print(len(result_fit2))
print(len(result_fit1))


df1['W'] = fit2(df1['abv'])*fit1(df1['jarakp'])

df1['total'] = df1['W']*df1['kernel']

df1['log']=np.log10(df1['total'])

rtest= np.linspace(0.24,5.5, 100)
vtest = np.linspace (0,1501,100)

D1_fit = fit1(rtest)
D2_fit = fit2(vtest)

plt.scatter(jarak2, norm, label='Data')
plt.plot(rtest, D1_fit, label='Fitting')
plt.ylabel('D/N')
plt.xlabel('Jarak Proyeksi')
plt.legend()

plt.scatter(vz, Nnorm, label='Data')
plt.plot(vtest,D2_fit, label='fitting')
plt.ylabel('D/N')
plt.xlabel('v (km/s)')
plt.legend()


-------Bobot Dinamikal Total----------
R = np.linspace(0, 6, 100)
VV = np.linspace(-3000,3000,100)

[RR,VVV] = np.meshgrid(R,VV)

zz = fit1(RR)*fit2(np.abs(VVV))

plt.figure(figsize=(8, 6))
level = [0.1,0.2,0.3,0.4,0.5,0.6]
cs = plt.contourf(RR,VVV,zz, levels=16, cmap='cool', extend='both')
cbar = plt.colorbar(cs)
#plt.clabel(cs, inline=1, fontsize=10)
plt.xlabel('Jarak Proyeksi (Mpc)')
plt.ylabel('Vpec (km/s)')
plt.title('Plot Kontur Bobot Dinamikal')
#plt.xlim(0,3)

--------Bobot Ruang Fase----------
def gaus(t):
    return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)

n = len(df1)
stdj = 1.06 * df1['jarakp'].std() * n**(-1/5)
stdv = 1.06 * df1['vpec'].std() * n**(-1/5)

Aa = 0
for ii in range(len(df1)):
    Bb = 0
    for jj in range(len(df1)):
        X = (df1.at[ii, 'jarakp'] - df1.at[jj, 'jarakp']) / stdj
        Y = (df1.at[ii, 'vpec'] - df1.at[jj, 'vpec']) / stdv
        gauss = gaus(X) * gaus(Y)
        Bb += gauss
    hasil = Bb / (len(df1) * stdv * stdj)
    Aa += np.log10(hasil)
    print(Aa)

gamma = np.power(10, Aa / len(df1))
print(f"Gamma: {gamma}")

def f0 (x1, y1):
  Bb = 0
  for Jj in range(len(df1)):
    X = (x1 - df1.at[Jj, 'jarakp'])/stdj
    Y = (y1 - df1.at[Jj, 'vpec'])/stdv
    gauss = gaus(X)*gaus(Y) / (stdj*stdv)
    Bb = Bb + gauss
  hasil = Bb /len(df1)
  return hasil

def f (x2,y2):
  Cc = 0
  for Kk in range(len(df1)):
    W = (x2 - df1.at[Kk, 'jarakp'])/(stdj*(np.sqrt(gamma/f0(df1.at[Kk,'jarakp'], df1.at[Kk,'vpec']))))
    V = (y2 - df1.at[Kk, 'vpec'])/(stdv*(np.sqrt(gamma/f0(df1.at[Kk,'jarakp'],df1.at[Kk,'vpec']))))
    gauss = gaus(W)*gaus(V)/((stdj*(np.sqrt(gamma/f0(df1.at[Kk,'jarakp'], df1.at[Kk, 'vpec']))))*(stdv*(np.sqrt(gamma/f0(df1.at[Kk,'jarakp'], df1.at[Kk,'vpec'])))))
    Cc = Cc + gauss
  D_AKM = Cc / len(df1)
  return D_AKM

df1['pdf'] = np.log10(f(df1['jarakp'], df1['vpec']))

df1['kernel'] = f(df1['jarakp'] , df1['vpec'])

df1.describe()

R = np.linspace(-0.1,6,100)
VV = np.linspace(-2500,2500,100)

RR,VVV = np.meshgrid(R,VV)

zzz = f(RR,VVV)
zzz[zzz <= 0] = np.nan

z1 = np.log10(zzz)


plt.figure(figsize=(8, 6))
CS = plt.contour(R, VV, z1, levels = 20, cmap='cool', extend='both')
cbar = plt.colorbar(CS)
plt.scatter(df1['jarakp'], df1['vpec'])
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('jarak proyeksi (Mpc)')
plt.ylabel('Vpec(km/s)')
#plt.colorbar(CS, label='log10')
#plt.ylim(-1500, 2500)
#plt.xlim(-0.1, 4)
plt.title('Plot Kontur Bobot Ruang Fase')

-----------Bobot Total------------
h = np.log10(fit1(RR)*fit2(np.abs(VVV))*f(RR,VVV))

plt.subplots(figsize=(12,6))
lev = np.arange(-1490,-340, 50)
CS = plt.contour(RR,VVV,h, levels=lev/100,cmap='cool', extend='both')
plt.clabel(CS, inline=1, fontsize=10)
plt.scatter(df1['jarakp'],df1['vpec'],c=df1['log'],cmap='bwr')
plt.colorbar()
plt.xlabel('Jarak Proyeksi')
plt.ylabel('Vpec')
plt.title('Plot Kontur Bobot Total')
#plt.xlim(-0.01, 4)
plt.ylim(-2500,2800)
