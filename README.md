# Convolución, correlación y transformación
## Descipción

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




### Requisitos 
Pyton 3.9.0 ó superior
### Librerias
* wfdb
* numpy
* matplotlib
* scipy.stats
