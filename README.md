# Convolución, correlación y transformación
## Descripción

El objetivo de esta práctica de laboratorio fue aplicar los conceptos de convolución, correlación y la transformada de Fourier, los cuales permiten el análisis de señales. La convolución es una herramienta esencial para analizar la respuesta de sistemas lineales e invariantes en el tiempo. La correlación permite medir la similitud entre señales, por su parte la transformada de Fourier facilita entender el comportamiento de las señales en el dominio de la frecuencia.

## Convolución 

La convolución es una operación que combina dos funciones para describir la superposición entre ambas. Este proceso consiste en deslizar una función sobre la otra, multiplicar los valores de ambas funciones en todos los puntos de superposición y sumar los productos para generar una nueva función que representa la interacción entre las funciones originales.

Específicamente en el procesamiento de señales, la convolución es fundamental para estudiar y diseñar sistemas lineales de tiempo invariante , como pueden los filtros digitales. La salida de un sistema LTI es la convolución de la señal de entrada X[n] y la respuesta al impulso del sistema H[n] obteniendo la señal de salida y[n].

En primer lugar se realizo la convolucion de manera manual mediante sumatorias utilizando el número de cédula como la señal de entrada (X[n]) y el número de código estudiantil como el sistema o respuesta al impulso (H[n]) para hallar la señal resultante de la convolución, esta operación se realizó para cada uno de los 3 integrantes del grupo 

![ConvAnd](https://github.com/user-attachments/assets/08f80e8e-7005-491d-87a0-ba36bcd23db8)

![ConvJo](https://github.com/user-attachments/assets/1ea15b51-cc0f-4227-86b1-75d4725b6847)

![ConvAB](https://github.com/user-attachments/assets/235d919e-7373-432d-803d-43ba8aa5b2b6)

Para realizar la convolución mediante funciones de python se desarrolló el siguiente código. En el cual se definen  dos arreglos para las secuencias h y x para cada persona.
Siendo h  la respuesta al impulso del sistema, que simula cómo reacciona un sistema a una señal de entrada y x es la señal de entrada, que representa la información que entra en el sistema
```python
Andrea_h = np.array([5, 6, 0, 0, 6, 8, 8])  
Andrea_x = np.array([1, 0, 0, 7, 3, 4, 0, 3, 9, 0])

Andrea_y = np.convolve(Andrea_x, Andrea_h)

axs[0].stem(Andrea_y, basefmt=" ", linefmt="b-", markerfmt="bo")
axs[0].set_title("Convolución Andrea")
axs[0].set_xlabel("n")
axs[0].set_ylabel("Amplitud")
```

La función np.convolve de la libreria numpy realiza la operación matemática de convolución para combinar x[n] y h[n]. El resultado es una nueva señal y[n], que describe cómo interactúan x y h entre sí. Posteriormente se grafica con la función axs[0].stem  que grafica la convolución en formato de gráfico de tallo (donde cada punto es un tallo que indica la amplitud de la señal en cada índice n).

El código de python da como resultado

![conv](https://github.com/user-attachments/assets/e27350be-d313-40d5-a08b-7691127761be)


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


La Densidad Espectral es una medida que describe cómo se distribuye la energía de una señal en función de la frecuencia, es útil para detectar ruido, analizar componentes de la señal y mejorar la calidad del procesamiento.

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
