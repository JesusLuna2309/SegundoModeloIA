import pandas as pd                 # Importa pandas para manejar datos en formato DataFrame
import seaborn as sb               # Importa seaborn para visualización (aunque no se usa en este código)
from sklearn.linear_model import LinearRegression  # Importa el modelo de regresión lineal de scikit-learn

# Lee el archivo CSV "altura.csv" y carga los datos en un DataFrame llamado 'datos'
datos = pd.read_csv("altura.csv")

# Muestra información general del DataFrame: número de filas, columnas, tipos de datos, etc.
datos.info()

# Muestra las primeras 5 filas del DataFrame para echar un vistazo a los datos
datos.head()

# Define la variable objetivo (target) 'y' que será la altura de las personas
# Se extrae la columna "altura" como una Serie pandas
y = datos["altura"]

# Accedemos a las columnas "edad" y "peso" del DataFrame 'datos'.
# Al usar doble corchete [[]], estamos indicando que queremos un nuevo DataFrame que incluya solo estas dos columnas.
# Esto devuelve una estructura tabular 2D (filas y columnas), donde:
#  - Cada fila representa un registro o muestra distinta (una persona diferente).
#  - Cada columna representa una característica (feature) que usaremos para predecir.
# Es fundamental que sea un DataFrame (2D) y no una Serie (1D) porque:
#  - El modelo de regresión lineal de scikit-learn espera una matriz de características en formato 2D para poder manejar múltiples variables independientes.
#  - Si usáramos un solo corchete, por ejemplo datos["edad"], obtendríamos una Serie 1D que solo contiene una columna, y el modelo no entendería que hay múltiples características.
# Por ejemplo, si 'datos' tiene 100 filas, 'x' será un DataFrame de forma (100, 2) con las columnas "edad" y "peso".
x = datos[["edad", "peso"]]


# Convierte las variables predictoras y la variable objetivo a arrays numpy
# Esto es necesario para que scikit-learn pueda trabajar con los datos
xProcesada = x.values    # Array 2D con las columnas 'edad' y 'peso'
yProcesada = y.values    # Array 1D con la columna 'altura'

# Crea una instancia del modelo de regresión lineal
modelo = LinearRegression()

# Entrena (ajusta) el modelo con las variables predictoras y la variable objetivo
# El modelo aprenderá la relación entre edad, peso y altura
modelo.fit(xProcesada, yProcesada)

# Define una muestra nueva para hacer una predicción: edad = 18 años, peso = 150 kg
peso = 150
edad = 18

# Usa el modelo entrenado para predecir la altura con la edad y peso dados
# Se pasa la muestra nueva en formato 2D: [[edad, peso]]
prediccion = modelo.predict([[edad, peso]])

# Imprime la predicción resultante con formato legible
print(f"Para una edad de {edad} y un peso de {peso} kg se estima que medira {prediccion[0]:.2f} cm")

# Evaluación del modelo:
# Calcula el coeficiente de determinación R² usando los datos de entrenamiento
# R² indica qué tan bien el modelo explica la variabilidad de la altura con las variables usadas
# Un valor cercano a 1.0 indica un buen ajuste; cercano a 0 indica mal ajuste
score = modelo.score(xProcesada, yProcesada)
print(f"Coeficiente de determinación R² del modelo: {score:.4f}")
