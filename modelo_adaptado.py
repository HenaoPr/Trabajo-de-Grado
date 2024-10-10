import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans

# Cargar los datos originales para entrenar los modelos
data = pd.read_csv("Data_final.csv")

# Preparación de los datos (sin la columna 'Career')
X = data.iloc[:, 0:-1]

# Normalización de los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Entrenamos DBSCAN
dbscan = DBSCAN()
data['Groups_DBSCAN'] = dbscan.fit_predict(X_scaled)

# Entrenamos KMeans
kmeans = KMeans(n_clusters=5, n_init=10)
data['Groups_KMeans'] = kmeans.fit_predict(X_scaled)

# Función para predecir el grupo con DBSCAN
def predecir_dbscan(test_data):
    test_scaled = scaler.transform(test_data)  # Normalizamos las respuestas del test
    grupo = dbscan.fit_predict(test_scaled)
    return grupo[0]

# Función para predecir el grupo con KMeans
def predecir_kmeans(test_data):
    test_scaled = scaler.transform(test_data)  # Normalizamos las respuestas del test
    grupo = kmeans.predict(test_scaled)
    return grupo[0]

# Mapeo de grupos a carreras (ajústalo según tus necesidades)
def obtener_carrera_por_grupo(grupo):
    grupos_a_carreras = {
        0: "Ingeniería",
        1: "Ciencias Sociales",
        2: "Arte",
        3: "Ciencias Exactas",
        4: "Negocios"
    }
    return grupos_a_carreras.get(grupo, "Carrera no encontrada")

# Función principal que decide qué modelo usar y devuelve la carrera
def predecir_carrera(test_data, metodo="kmeans"):
    if metodo == "dbscan":
        grupo = predecir_dbscan(test_data)
    else:
        grupo = predecir_kmeans(test_data)

    carrera = obtener_carrera_por_grupo(grupo)
    return carrera
