# TFM

En este repositorio está todo el código usado en el Trabajo de Fin de Máster "Predicción de las Curvas de Oferta de Electricidad del Mercado Diario". Los datos usados están disponibles en este [enlace](https://www.omie.es/es/file-access-list).

- ***lstm.py*** y ***lstm_hidden.py*** son las arquitecturas de las redes LSTM y LSTM con *hidden state*.
- ***utils.py*** contiene algunas funciones útiles, como el preprocesado de los datos, calcular la integral del valor absoluto de la resta de dos funciones escalonadas y visualizar las predicciones de los modelos.
- ***raw_dataset_and_grids.ipynb*** es un *notebook* con el preprocesado inicial de los datos y la creación de las mallas equiespaciada y no equiespaciada y sus ofertas asociadas.
- ***user_guide.ipynb*** es un *notebook* que contiene un ejemplo del entrenamiento de cada modelo y un ejemplo de la carga de los modelos una vez se hayan guardado los pesos. También se muestra cómo realizar predicciones sobre un conjunto de días.
