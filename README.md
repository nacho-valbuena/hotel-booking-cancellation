# Predicción de Cancelaciones de Reservas Hoteleras

Proyecto final de **Data Science II #90420** desarrollado por **José Valbuena**.

## Descripción del proyecto

Las cancelaciones de reservas generan incertidumbre para los establecimientos hoteleros, ya que pueden afectar la planificación de la ocupación, la disponibilidad de habitaciones y la gestión de ingresos.

El objetivo de este proyecto es desarrollar un modelo de **Machine Learning** capaz de estimar la probabilidad de que una reserva hotelera sea cancelada utilizando únicamente información que pueda conocerse antes de observar el resultado final de la reserva.

El proyecto recorre un pipeline completo de Data Science: auditoría y limpieza de datos, análisis exploratorio, ingeniería de atributos, prevención de *data leakage*, entrenamiento y comparación de modelos, optimización de hiperparámetros, evaluación final e interpretación mediante SHAP.

---

## Problema de Machine Learning

El problema corresponde a una tarea de **clasificación supervisada binaria**.

La variable objetivo es:

```text
is_canceled
```

donde:

- `0`: la reserva no fue cancelada.
- `1`: la reserva fue cancelada.

La pregunta principal del proyecto es:

> **¿Es posible predecir si una reserva hotelera será cancelada a partir de las características conocidas de la reserva?**

La salida del modelo se interpreta principalmente como una **probabilidad de cancelación**, permitiendo utilizarla como un sistema de priorización o *ranking* de riesgo.

---

## Dataset

El archivo utilizado es:

```text
hotel_bookings.csv
```

El conjunto original contiene:

- **119.390 observaciones**
- **32 variables**
- **75.166 reservas no canceladas**
- **44.224 reservas canceladas**

Distribución aproximada de la variable objetivo:

| Clase | Porcentaje |
|---|---:|
| No cancelada | 62,96 % |
| Cancelada | 37,04 % |

Durante la limpieza básica se eliminaron únicamente:

- 180 registros sin huéspedes registrados.
- 1 registro con `adr` negativo.

El conjunto utilizado posteriormente contiene **119.209 observaciones**.

### Fuente del dataset

Kaggle, **Hotel Booking Demand**:

https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand?resource=download

Si el dataset no se incluye en el repositorio, debe descargarse desde la fuente anterior y colocarse en la ubicación indicada en la sección de ejecución.

---

## Auditoría y limpieza de datos

Durante la inspección inicial se analizaron:

- dimensiones y tipos de datos;
- cardinalidad de las variables;
- valores faltantes;
- duplicados;
- distribuciones numéricas y categóricas;
- valores extremos;
- observaciones potencialmente inconsistentes.

### Valores faltantes

Se detectaron valores faltantes principalmente en:

- `company`
- `agent`
- `country`
- `children`

Las decisiones principales fueron:

- `children`: imputación mediante la **moda**, debido a la muy baja cantidad de valores faltantes y al predominio de la categoría 0.
- `country`: reemplazo de faltantes por la categoría `Unknown`.
- `agent` y `company`: no se imputaron como magnitudes numéricas, ya que representan identificadores.

### Duplicados

Se detectaron **31.994 filas exactamente iguales**.

No se eliminaron automáticamente porque el dataset no posee un identificador único de reserva que permita confirmar que se trate de registros duplicados erróneos. Además, eliminar estas observaciones modifica considerablemente la proporción de cancelaciones.

Para reducir el riesgo de obtener métricas artificialmente optimistas, la separación entre entrenamiento y prueba se realizó considerando grupos de observaciones con predictores idénticos.

---

## Prevención de Data Leakage

Uno de los objetivos metodológicos principales fue evitar utilizar información que no estaría disponible en el momento en que se desea realizar la predicción.

Se excluyeron del modelado:

```text
reservation_status
reservation_status_date
assigned_room_type
booking_changes
days_in_waiting_list
```

También se excluyeron los identificadores originales:

```text
agent
company
```

y se utilizaron variables derivadas más interpretables cuando correspondía.

---

## Análisis Exploratorio de Datos

El EDA incluyó:

- distribución de la variable objetivo;
- histogramas y boxplots de variables numéricas;
- distribuciones de variables categóricas;
- comparación de tasas de cancelación entre categorías;
- análisis de correlaciones;
- estudio de valores faltantes y valores atípicos;
- relación entre diferentes características y `is_canceled`.

El análisis mostró que el riesgo de cancelación no depende de una única variable, sino de una combinación de factores relacionados con:

- anticipación de la reserva;
- condiciones comerciales;
- segmento de mercado;
- historial del cliente;
- solicitudes especiales;
- características operativas de la reserva.

---

## Ingeniería de atributos

Se generaron variables derivadas para representar de forma más directa algunas características de las reservas.

Entre ellas se incluyen variables relacionadas con:

- cantidad total de noches;
- cantidad total de huéspedes;
- presencia de niños;
- tipo de grupo o familia;
- presencia de agente;
- presencia de empresa;
- historial de reservas previas;
- proporción de cancelaciones anteriores.

El preprocesamiento se integró mediante herramientas de `scikit-learn`, utilizando `Pipeline` y `ColumnTransformer` para mantener un flujo reproducible.

Las variables categóricas se procesan mediante **One-Hot Encoding**, mientras que las transformaciones numéricas se aplican dentro del pipeline para evitar contaminación entre entrenamiento y evaluación.

---

## Estrategia de validación

El conjunto de prueba se mantuvo aislado durante:

- comparación inicial de modelos;
- selección del candidato;
- optimización de hiperparámetros;
- elección del umbral de clasificación.

La validación se realizó mediante **validación cruzada por grupos**, evitando que observaciones con la misma combinación de predictores aparezcan simultáneamente en entrenamiento y validación.

La métrica principal para la selección de modelos fue **ROC-AUC**.

También se analizaron:

- PR-AUC;
- Precision;
- Recall;
- F1-score;
- Accuracy;
- matriz de confusión.

---

## Modelos evaluados

Se compararon los siguientes modelos:

1. `DummyClassifier`
2. Regresión Logística
3. Árbol de Decisión
4. Random Forest
5. XGBoost

Resultados medios en validación cruzada:

| Modelo | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0,9268** | 0,8973 | 0,8304 | 0,7401 | **0,7826** |
| Random Forest | 0,9254 | **0,8981** | **0,8702** | 0,6836 | 0,7656 |
| Árbol de decisión | 0,9113 | 0,8671 | 0,8000 | **0,7440** | 0,7707 |
| Regresión logística | 0,8993 | 0,8666 | 0,8145 | 0,6727 | 0,7369 |
| Dummy | 0,5000 | 0,3742 | 0,0000 | 0,0000 | 0,0000 |

**XGBoost** fue seleccionado para la etapa de optimización por presentar el mayor ROC-AUC y un buen equilibrio entre las métricas complementarias.

---

## Optimización de hiperparámetros

La optimización se realizó mediante `RandomizedSearchCV`, utilizando exclusivamente el conjunto de entrenamiento.

El mejor resultado obtenido fue:

```text
ROC-AUC medio en validación cruzada = 0,9316
```

Mejores hiperparámetros:

```python
{
    "subsample": 1.0,
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "colsample_bytree": 1.0
}
```

---

## Selección del umbral

Además de evaluar la calidad de las probabilidades mediante ROC-AUC, se estudió el efecto de diferentes umbrales de clasificación utilizando predicciones *out-of-fold* del conjunto de entrenamiento.

| Umbral | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0,30 | 0,7174 | **0,8862** | 0,7929 |
| **0,40** | 0,7815 | 0,8223 | **0,8014** |
| 0,50 | 0,8321 | 0,7600 | 0,7944 |
| 0,60 | 0,8817 | 0,6775 | 0,7662 |
| 0,70 | **0,9366** | 0,5650 | 0,7048 |

Se seleccionó un **umbral de 0,40**, buscando aumentar la detección de reservas que efectivamente se cancelarán sin deteriorar excesivamente la precisión.

---

## Resultados finales

El modelo final fue evaluado una sola vez sobre el conjunto de prueba aislado.

| Métrica | Resultado |
|---|---:|
| **Accuracy** | **0,8501** |
| **ROC-AUC** | **0,9298** |
| **PR-AUC** | **0,8949** |
| **Precision** | **0,7783** |
| **Recall** | **0,8109** |
| **F1-score** | **0,7943** |
| **Umbral utilizado** | **0,40** |

El ROC-AUC obtenido muestra una elevada capacidad para ordenar reservas según su riesgo de cancelación.

El Recall de aproximadamente **81 %** indica que el modelo identifica una proporción importante de las reservas que finalmente se cancelan.

---

## Interpretabilidad con SHAP

Se utilizó **SHAP** para estudiar la contribución de las características a las predicciones del modelo.

El análisis global se realizó sobre una muestra de 1.000 observaciones del conjunto de prueba.

Principales características según la importancia SHAP media absoluta:

| Característica | Mean \|SHAP\| |
|---|---:|
| `country_PRT` | 0,8517 |
| `deposit_type_Non Refund` | 0,8062 |
| `market_segment_Online TA` | 0,5387 |
| `required_car_parking_spaces` | 0,4546 |
| `total_of_special_requests` | 0,4494 |
| `lead_time` | 0,4217 |
| `deposit_type_No Deposit` | 0,3135 |
| `previous_cancel_rate` | 0,3099 |
| `arrival_date_year` | 0,2395 |
| `customer_type_Transient` | 0,1770 |

Estas importancias deben interpretarse como **influencia predictiva dentro del modelo**, no como evidencia de causalidad.

Además del análisis global, el notebook incluye una explicación SHAP local para mostrar cómo distintas características aumentan o disminuyen el riesgo estimado para una reserva individual.

---

## Principales conclusiones de negocio

Los resultados indican que el riesgo de cancelación surge de la combinación de múltiples factores.

Entre los patrones más relevantes observados durante el análisis se encuentran:

- las reservas realizadas con mayor anticipación tienden a presentar mayor riesgo de cancelación;
- los diferentes tipos de depósito presentan comportamientos de cancelación claramente diferenciados;
- el segmento de mercado aporta información predictiva importante;
- el historial de cancelaciones previas contribuye a estimar el riesgo futuro;
- las solicitudes especiales y los espacios de estacionamiento solicitados aparecen asociados con diferencias en el comportamiento de cancelación.

El modelo puede utilizarse como un **ranking de riesgo** en lugar de limitarse a una clasificación rígida.

Las probabilidades estimadas podrían apoyar decisiones como:

- priorizar el seguimiento de reservas de alto riesgo;
- mejorar la planificación de ocupación;
- apoyar estrategias de *overbooking*;
- evaluar políticas diferenciadas de depósitos;
- anticipar disponibilidad potencial de habitaciones.

Estas aplicaciones deben validarse posteriormente con información real sobre costos, políticas comerciales y operación del hotel.

---

## Limitaciones

El proyecto presenta algunas limitaciones que deben considerarse al interpretar los resultados:

1. El dataset no contiene un identificador único de reserva, lo que dificulta confirmar si las filas idénticas son duplicados reales.
2. Algunas variables fueron excluidas por riesgo de *data leakage*, por lo que el escenario predictivo adoptado es deliberadamente conservador.
3. La interpretación SHAP describe el comportamiento del modelo y no relaciones causales.
4. El modelo fue evaluado sobre el conjunto de datos disponible; su desempeño en otros hoteles, períodos o mercados requiere validación externa.
5. El umbral de 0,40 fue seleccionado buscando un equilibrio entre Precision y Recall. En una implementación real debería ajustarse según el costo económico de falsos positivos y falsos negativos.

---

## Estructura del repositorio

```text
hotel-booking-cancellation/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── Entrega_Final_Jose_Valbuena.ipynb
│
├── models/
│   └── hotel_booking_cancellation_model.joblib
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Instalación

Se recomienda utilizar un entorno virtual.

```bash
python -m venv .venv
```

Activación en Windows:

```bash
.venv\Scripts\activate
```

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

Versiones principales utilizadas durante la ejecución final:

| Herramienta | Versión |
|---|---|
| Python | 3.13.5 |
| pandas | 2.2.3 |
| numpy | 2.3.5 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| scikit-learn | 1.8.0 |
| xgboost | 3.1.3 |
| shap | 0.50.0 |
| joblib | 1.5.3 |

---

## Ejecución

1. Clonar o descargar el repositorio.
2. Crear y activar el entorno de Python.
3. Instalar las dependencias con `pip install -r requirements.txt`.
4. Descargar `hotel_bookings.csv` desde la fuente indicada.
5. Colocar el dataset en la ruta esperada por el notebook.
6. Abrir:

```text
notebooks/Entrega_Final_Jose_Valbuena.ipynb
```

7. Ejecutar todas las celdas desde el comienzo.

El notebook utiliza semillas aleatorias (`random_state=42`) en las etapas principales para facilitar la reproducción de los resultados.

---

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- SHAP
- Joblib
- Jupyter Notebook / Google Colab

---

## Archivos principales

### Notebook

```text
notebooks/Entrega_Final_Jose_Valbuena.ipynb
```

Contiene el pipeline completo desde la carga y auditoría de datos hasta la evaluación e interpretación del modelo final.

### Modelo entrenado

```text
models/hotel_booking_cancellation_model.joblib
```

### Dependencias

```text
requirements.txt
```

---

## Reproducibilidad

El proyecto fue diseñado para mantener las transformaciones y el modelo dentro de pipelines reproducibles.

Antes de la entrega definitiva se recomienda comprobar que:

- el repositorio sea público;
- el notebook se ejecute desde cero sin errores;
- el dataset o sus instrucciones de acceso estén disponibles;
- `requirements.txt` contenga todas las dependencias;
- la URL del repositorio pueda abrirse sin autenticación.

---

## Autor

**José Valbuena**  
Data Science II #90420