# Convolución, correlación y transformación
## Descripción

El objetivo de esta práctica de laboratorio fue aplicar los conceptos de convolución, correlación y la transformada de Fourier, los cuales permiten el análisis de señales. La convolución es una herramienta esencial para analizar la respuesta de sistemas lineales e invariantes en el tiempo. La correlación permite medir la similitud entre señales, por su parte la transformada de Fourier facilita entender el comportamiento de las señales en el dominio de la frecuencia.

## Convolución 


```python
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


```
### Señal en función del tiempo
Se obtuvo una señal de ECG de la base de datos Physionet. La base de datos contiene 310 registros obtenidos de 90 personas, que incluyen la derivación I registrada durante 20 segundos y digitalizada a 500 Hz. Cada registro incluye tanto la señal sin procesar como la señal filtrada.

Se guardaron los datos que fueron extraidos de los archivos .dat y .hea en un arreglo Numpy gracias a la libreria WFDB. De estos archivos tambien podemos obtener información sobre la señal como su frecuencia, número de muestras, valor mínimo y máximo de los datos.

```python
archivo = 'C:/Users/Usuario/Downloads/Señales lab 2/rec_1'
registro = wfdb.rdrecord(archivo)
arreglo = registro.p_signal 
datos = arreglo[:, 1]

frecuencia = registro.fs
print("Frecuencia de muestreo:",frecuencia, "Hz")
periodo = 1 / frecuencia
print("Tiempo entre muestras:",periodo, "s")
num_muestras = registro.sig_len
print("El número de muestras en el registro es:",num_muestras)
valor_min = np.min(datos)
valor_max = np.max(datos)
print("Valor mínimo:",valor_min)
print("Valor máximo:",valor_max)
```
Además se graficó la señal en función del tiempo y se describió en cuanto a su clasificación: Una señal de ECG es multicanal pero en este caso vemos la gráfica de solo la derivación I haciendola de un único canal, además es unidimensional, ya que depende del tiempo y discreta al estar digitalizada.

![Señal de ECG](https://github.com/user-attachments/assets/01cf75ec-e25a-4585-bbe4-6c3dc7a7b356)


## Estadísticos descriptivos
Para caracterizar la señal se calcularon la media, la desviación estandar, el coeficiente de variación y un histograma de frecuencias junto con la función de probabilidad. Estos estadísticos descriptivos permiten conocer que tan dispersos o juntos están los datos, asi como su distribución y frecuencia.
```python
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
```
## Transformada de Fourier y Densidad espectral
La Transformada de Fourier es una herramienta fundamental en el procesamiento de señales porque permite convertir una señal del dominio del tiempo al dominio de la frecuencia, permitiendo identificar qué frecuencias están presentes en una señal y con qué intensidad. Se utiliza la Transformada Rápida de Fourier (FFT) sobre la señal y además se gráfica su espectro de frecuencia hasta 50 Hz.

```python
fxt = np.fft.fft(datos)
frecuencias = np.fft.fftfreq(len(datos), d=1/frecuencia)
limite_frecuencia = 50
mask = (frecuencias >= 0) & (frecuencias <= limite_frecuencia)
plt.figure(figsize=(10, 5))

plt.plot(frecuencias[mask], np.abs(fxt[mask]))
plt.title("FFT")
plt.ylabel("Magnitud")
plt.xlabel("Frecuencia (Hz)")
plt.grid()
plt.xlim(0, limite_frecuencia)  
plt.show()
```
![FFT](https://github.com/user-attachments/assets/99e80079-0f2a-4836-a531-08630d3283f8)


La Densidad Espectral es una medida que describe cómo se distribuye la energía de una señal en función de la frecuencia, es útil para para detectar ruido, analizar componentes de la señal y mejorar la calidad del procesamiento.

```python
frecuencias_psd, psd = welch(datos, fs=frecuencia, nperseg=1024)
plt.figure(figsize=(8, 5))
plt.plot(frecuencias_psd, psd, color='blue')
plt.title("Densidad Espectral de Potencia (PSD)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia (V²/Hz)")
plt.grid()
plt.xlim(0, limite_frecuencia)
plt.show()
```
![densidad](https://github.com/user-attachments/assets/97d615d4-9a8d-43d4-a4f6-6eac5bfd0de0)



### Requisitos 
Pyton 3.9.0 ó superior
### Librerias
* wfdb
* numpy
* matplotlib
* scipy.stats
