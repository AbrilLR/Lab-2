import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation, norm
from scipy.fft import fft, fftfreq
from scipy.signal import welch


Andrea_h = np.array([5, 6, 0, 0, 6, 8, 8])  
Andrea_x = np.array([1, 0, 0, 7, 3, 4, 0, 3, 9, 0])  

Johan_h = np.array([5, 6, 0, 0, 7, 0 , 2])  
Johan_x = np.array([1, 0, 7, 5, 6, 5, 1, 2, 3, 3])  

Abril_h = np.array([5, 6, 0, 0, 7, 2, 9])  
Abril_x = np.array([1, 1, 0, 4, 5, 4, 4, 8, 8, 8])


Andrea_y = np.convolve(Andrea_x, Andrea_h)
Johan_y = np.convolve(Johan_x, Johan_h)
Abril_y = np.convolve(Abril_x, Abril_h)

print("Andrea (h[n] * x[n]):", Andrea_y .tolist())
print("Johan (h[n] * x[n]):", Johan_y.tolist())
print("Abril (h[n] * x[n]):", Abril_y.tolist())


fig, axs = plt.subplots(3, 1, figsize=(12, 8))

axs[0].stem(Andrea_y, basefmt=" ", linefmt="b-", markerfmt="bo")
axs[0].set_title("Convolución Andrea")
axs[0].set_xlabel("n")
axs[0].set_ylabel("Amplitud")

axs[1].stem(Johan_y, basefmt=" ", linefmt="r-", markerfmt="ro")
axs[1].set_title("Convolución Johan")
axs[1].set_xlabel("n")
axs[1].set_ylabel("Amplitud")

axs[2].stem(Abril_y, basefmt=" ", linefmt="g-", markerfmt="go")
axs[2].set_title("Convolución Abril")
axs[2].set_xlabel("n")
axs[2].set_ylabel("Amplitud")

plt.tight_layout()
plt.show()

fs = 1 / 1.25e-3
Ts = 1 / fs 
n = np.arange(0, 9)  


x1 = np.cos(2 * np.pi * 100 * n * Ts)
x2 = np.sin(2 * np.pi * 100 * n * Ts)

# correlación 
correlacion = np.correlate(x1, x2, mode='full')
correlacion_n = correlacion / (np.linalg.norm(x1) * np.linalg.norm(x2))

print("correlación:", np.max(correlacion_n))
print("representación secuencial")
for i in range(len(n)):
    print(f"n={n[i]}, x1[n]={x1[i]:.3f}, x2[n]={x2[i]:.3f}")

# Gráfica de señales
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.stem(n, x1, linefmt='b-', markerfmt='bo', basefmt=" ")
plt.stem(n, x2, linefmt='r-', markerfmt='ro', basefmt=" ")
plt.title("Señales x1[n] (azul) y x2[n] (rojo)")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid()


# Gráfica de correlación
lags = np.arange(-len(n) + 1, len(n))
plt.subplot(2, 1, 2)
plt.stem(lags, correlacion_n, linefmt='g-', markerfmt='go', basefmt=" ")
plt.title("Correlación cruzada")
plt.xlabel("Desplazamiento k")
plt.ylabel("Correlación")
plt.grid()

plt.tight_layout()
plt.show()


archivo = 'C:\\Users\\sebas\\OneDrive\\Escritorio\\Nueva carpeta\\Señales lab 2_Tercer punto\\rec_1'

registro = wfdb.rdrecord(archivo)
arreglo = registro.p_signal 
datos = arreglo[:, 1] 

frecuencia = registro.fs
print("Frecuencia de muestreo:",frecuencia, "Hz")
periodo = 1 / frecuencia
print("Tiempo entre muestras:",periodo, "s")

num_muestras = registro.sig_len

print(f"El número de muestras en el registro es: {num_muestras}")

valor_min = np.min(datos)
valor_max = np.max(datos)

print(f"Valor mínimo: {valor_min}")
print(f"Valor máximo: {valor_max}")

#Gráfica
tiempo = np.arange(0, len(datos))*periodo
plt.figure(figsize=(10, 6))
plt.plot(tiempo, datos)
plt.title("Señal ECG")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje(mV)")
plt.grid(True)
plt.show()

#Media
media = np.mean(datos)
print("Media:",media)

#Desviación 
desv = np.std(datos)
print("Desviación:",desv)

#Coeficiente de variación
cv = variation(datos) * 100
print("Coeficiente de variación:",cv,"%")

#Histograma
plt.figure(figsize=(10, 6))
count, bins, _ = plt.hist(datos, bins=17, alpha=0.6, color='g', edgecolor='black', label="Histograma (Frecuencia)")
x = np.linspace(min(datos), max(datos), 900000)
y = norm.pdf(x, media, desv)  

scale_factor = max(count) / max(y)  
y_scaled = y * scale_factor

plt.plot(x, y_scaled, color='red', linewidth=2, label="Función de probabilidad")

plt.title('Histograma de Frecuencia y Función de probabilidad ECG')
plt.xlabel('Voltaje (mV)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.legend()
plt.show()

#Fourier
fxt = np.fft.fft(datos)
frecuencias = np.fft.fftfreq(len(datos), d=1/frecuencia)

limite_frecuencia = 50
mask = (frecuencias >= 0) & (frecuencias <= limite_frecuencia)

plt.figure(figsize=(10, 5))

# Magnitud de la FFT
plt.plot(frecuencias[mask], np.abs(fxt[mask]))
plt.title("FFT")
plt.ylabel("Magnitud")
plt.xlabel("Frecuencia (Hz)")
plt.grid()
plt.xlim(0, limite_frecuencia)  
plt.show()

#Densidad espectral de potencia
frecuencias_psd, psd = welch(datos, fs=frecuencia, nperseg=1024)
plt.figure(figsize=(8, 5))
plt.plot(frecuencias_psd, psd, color='blue')
plt.title("Densidad Espectral de Potencia (PSD)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia (V²/Hz)")
plt.grid()
plt.xlim(0, limite_frecuencia)
plt.show()

#Media de la frecuencia
frecuencia_media = np.sum(frecuencias_psd * psd) / np.sum(psd)
print("Frecuencia media:",frecuencia_media,"Hz")

#Mediana de la frecuencia
potencia_acumulada = np.cumsum(psd) 
potencia_total = np.sum(psd)
frecuencia_mediana = frecuencias_psd[np.where(potencia_acumulada >= potencia_total / 2)[0][0]]
print("Frecuencia mediana:",frecuencia_mediana,"Hz")

# Desviación estándar de la frecuencia
desviacion_frecuencia = np.sqrt(np.sum(psd * (frecuencias_psd - frecuencia_media)**2) / np.sum(psd))
print("Desviación estándar de la frecuencia:", desviacion_frecuencia,"Hz")

# Histograma de la magnitud de la FFT
plt.figure(figsize=(10, 6))
plt.hist(np.abs(fxt[mask]), bins=17, color='b', alpha=0.7, edgecolor='black')
plt.title("Histograma de la Magnitud de la Transformada de Fourier")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Frecuencia")
plt.grid()
plt.show()




