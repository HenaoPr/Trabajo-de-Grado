from flask import Flask, render_template, request
import pandas as pd  # Importación de pandas
from modelo_adaptado import predecir_carrera  # Asegúrate de importar la función que predice la carrera
  # Asegúrate de importar la función que predice la carrera

app = Flask(__name__)

@app.route('/')
def index():
    # Renderizar la página principal donde se encuentra el formulario del test
    return render_template('test.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Obtener las respuestas del formulario (120 preguntas)
        respuestas = [float(request.form.get(f'question{i}')) for i in range(1, 121)]

        # Verificar que tenemos exactamente 120 respuestas
        if len(respuestas) != 120:
            return "Error: Se esperaban 120 respuestas pero se recibieron {}".format(len(respuestas)), 400

        # Nombres de las columnas originales (excluyendo 'Career')
        columnas_originales = ['O_score', 'C_score', 'E_score', 'A_score', 'N_score', 
                               'Numerical Aptitude', 'Spatial Aptitude', 'Perceptual Aptitude', 
                               'Abstract Reasoning', 'Verbal Reasoning']

        # Agrupar las 120 respuestas en 10 grupos (promediando cada grupo de 12 preguntas)
        respuestas_agrupadas = [
            sum(respuestas[i:i+12]) / 12 for i in range(0, 120, 12)
        ]

        # Verificar que solo hay 10 valores en respuestas_agrupadas
        if len(respuestas_agrupadas) != 10:
            return "Error: No se pudo agrupar correctamente las respuestas. Se obtuvieron {} grupos en lugar de 10".format(len(respuestas_agrupadas)), 400

        # Convertir las respuestas agrupadas en un DataFrame con los nombres de columnas originales
        test_data = pd.DataFrame([respuestas_agrupadas], columns=columnas_originales)

        # Mostrar el DataFrame (para depuración)
        print("DataFrame generado:", test_data)

        # Llamar a la función que predice la carrera (sin incluir la columna 'Career')
        carrera = predecir_carrera(test_data, metodo="kmeans")

        # Mostrar el resultado en la página de resultados
        return render_template('result.html', carrera=carrera)


if __name__ == '__main__':
    app.run(debug=True)
