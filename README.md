# Detección y Rastreo de Objetos en Tiempo Real — Hack Knights IA Challenge 2026

Solución de visión artificial para el **Reto Mechatronics AI**. Detecta, localiza y clasifica cinco objetos físicos a partir de video en vivo, usando **YOLO (Ultralytics)** con un modelo ajustado por el equipo (`best.pt`), **OpenCV** para la captura y la interfaz, y **ByteTrack** para asignar un ID a cada objeto entre frames. Funciona en **CPU** (no requiere GPU).

**Equipo:** `Los Transistonicos'

---

## 1. Clases reconocidas

| Clase del modelo | Nombre en pantalla |
|------------------|--------------------|
| `cup`            | Taza               |
| `bottle`         | Botella            |
| `scissors`       | Tijeras            |
| `screwdriver`    | Destornillador     |
| `cell phone`     | Celular            |

Cualquier objeto que no pertenezca a estas cinco clases **es ignorado** (no se dibuja caja ni se le asigna una clase oficial).

---

## 2. Características (mapa contra los criterios del reto)

| Criterio | Cómo se cumple |
|----------|----------------|
| Captura de video en vivo con fuente configurable | `cv2.VideoCapture` con el índice de cámara editable (ver sección 5). Compatible con cámara de laptop o celular. |
| Cajas delimitadoras | Se dibuja un rectángulo verde por detección. |
| Clase y confianza visibles | Etiqueta `#ID Clase: XX.X%` sobre cada caja. |
| Varios objetos simultáneos | Se procesan todas las cajas del frame en cada iteración. |
| FPS visibles y mínimo de 15 en CPU | Panel en pantalla con "FPS Real" (verde si ≥ 15, rojo si < 15), suavizado con media móvil exponencial. |
| Umbral de confianza configurable | Barra deslizante **"Confianza (%)"** (limitada internamente entre 10 % y 95 %). |
| Cierre limpio con `q` | Tecla `q` o cierre de la ventana; los recursos se liberan en el bloque `finally`. |
| Manejo de error sin cámara / conexión perdida | Mensaje de error si no abre la cámara; si fallan más de 60 lecturas seguidas, el programa termina de forma controlada. |
| Técnica contra falsos positivos | Ver sección 6. |
| **Bonus:** contador por clase | Panel con el conteo de objetos por clase en pantalla. |
| **Bonus:** seguimiento con ID | ByteTrack (`bytetrack.yaml`) con `persist=True`. |
| **Bonus:** ONNX / embebido | `[COMPLETAR si se exportó: comando de exportación y resultados]` |

---

## 3. Estructura del repositorio

```
.
├── main.py              # Detección y rastreo en vivo
├── best.pt              # Pesos del modelo ajustado por el equipo
├── requirements.txt     # Dependencias con versiones
├── README.md            # Este archivo
├── respuestas.md        # Decisiones técnicas del equipo
├── metricas.md          # Reporte de métricas (precision, recall, mAP, matriz de confusión)
└── dataset/
    ├── data.yaml
    ├── train/           # images/ y labels/
    ├── val/             # images/ y labels/
    └── test/            # Conjunto de prueba SEPARADO (no usado para entrenar ni ajustar)
```


---

## 4. Instalación

**Requisitos:** Python 3.10 a 3.12, una cámara web (integrada o de celular) y Windows (ver "Limitaciones").

```bash
# 1. Clonar o descomprimir el proyecto
git clone <URL-DEL-REPOSITORIO>
cd <carpeta-del-proyecto>

# 2. Crear y activar un entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell / CMD)

# 3. Instalar dependencias
pip install -r requirements.txt
```

Contenido sugerido de `requirements.txt`
```
ultralytics==<8.4.175>
opencv-python==<5.0.0.93>
torch==<2.14.1>
numpy==<2.5.3>
```

Para instalar PyTorch solo con CPU (más ligero):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 5. Configuración de la cámara y ejecución

### Cámara de la laptop
Por defecto se usa la cámara con índice `0`. Si tienes varias cámaras, cambia el índice en `main.py`:

```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # 0 = cámara integrada, 1 = siguiente cámara, etc.
```

### Cámara de celular
Usa una aplicación que exponga el celular como webcam (por ejemplo **DroidCam**, **Iriun Webcam** o **Camo**):

1. Instala la app en el celular y el cliente en la computadora.
2. Conecta ambos (USB o Wi-Fi en la misma red).
3. La cámara del celular aparece como una cámara más: prueba con el índice `1` (o `2`) en `cv2.VideoCapture(...)`.
4. Verifica que la imagen se vea antes de la demostración.

### Ejecutar

Asegúrate de que `best.pt` esté **en la misma carpeta que `main.py`**:

```bash
python main.py
```

### Controles

| Control | Función |
|---------|---------|
| Barra **Confianza (%)** | Umbral mínimo de confianza (por defecto 50 %). |
| Barra **Limite FPS** | Tope de FPS del ciclo (por defecto 20, mínimo 5). |
| Tecla **`q`** | Cierra el programa de forma limpia. |
| Cerrar la ventana | También termina el programa correctamente. |

### Configuración principal (parte superior de `main.py`)

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `MODELO` | `best.pt` | Ruta del modelo (relativa al script). |
| `IMGSZ` | `256` | Tamaño de entrada de inferencia; menor tamaño = más FPS. |
| `CAM_W`, `CAM_H` | `640`, `480` | Resolución de captura. |
| `FPS_MIN_SLIDER` | `5` | Mínimo permitido en la barra de límite de FPS. |

---

## 6. Técnica contra falsos positivos

Se combinan tres mecanismos:

1. **Filtro por clase.** La inferencia se restringe a los IDs de las cinco clases oficiales (`classes=ids_interes`). Cualquier objeto ajeno se ignora y nunca recibe una clase oficial.
2. **Umbral de confianza configurable.** Se parte de 50 % y puede ajustarse en vivo según la iluminación y el fondo. Subirlo reduce falsos positivos; bajarlo mejora la detección de objetos parcialmente ocultos.
3. **Rastreo con ByteTrack.** Al mantener un ID por objeto entre frames, las detecciones se vuelven más estables y se puede distinguir un objeto real de una detección esporádica.

**Justificación:** el reto evalúa objetos ajenos a las clases oficiales y fondo vacío. El filtro por clase evita clasificar mal objetos desconocidos, y el umbral configurable permite adaptarse a las condiciones del jurado sin reiniciar el programa. Con un modelo pequeño y resolución baja (`IMGSZ = 256`) para llegar a 15 FPS en CPU, es esperable más ruido; el umbral en vivo compensa ese compromiso.

---

## 7. Modelo, datos y entrenamiento


- **Modelo base:**  YOLOv8n 
- **Tipo de ajuste:** ajuste fino con dataset propio
- **Herramientas:**  Google Colab, Roboflow
- **Dataset propio:**400imágenes en total, capturadas con la cámara del equipo, mínimo 20 por clase, con variaciones de iluminación, fondo y posición. Etiquetas en formato YOLO dentro de `dataset/`.
- **Etiquetado:**  Roboflow, una caja por objeto
- **Hiperparámetros principales:**

| Parámetro | Valor |
|-----------|-------|
| Épocas |     280 |
| Tamaño de imagen (entrenamiento) | `[COMPLETAR]` |
| Batch size | `[COMPLETAR]` |
| Optimizador / learning rate | `[COMPLETAR]` |
| Partición train / val / test | `[COMPLETAR]` |

- **Conjunto de prueba:** `dataset/test/` está separado del usado para entrenar y ajustar.

### Fuentes externas y créditos

- Ultralytics YOLO — https://github.com/ultralytics/ultralytics
- Pesos preentrenados  yolov8n.pt / yolo11n.pt (Ultralytics)
- ByteTrack (integrado en Ultralytics) — Zhang et al., 2022
- OpenCV — https://opencv.org
- Tutoriales: https://youtu.be/r0RspiLG260?si=GZflcKi-80Mj8NXz
https://youtu.be/z70ZrSZNi-8?si=KtdCL1pZNuNJc2kk

---

## 8. Evaluación y métricas

Las métricas se calculan sobre el conjunto de prueba separado (`dataset/test/`) con el comando de validación de Ultralytics:

```bash
yolo detect val model=best.pt data=dataset/data.yaml split=test imgsz=256 plots=True
```

Esto genera precision, recall, mAP@0.5, mAP@0.5:0.95 y la matriz de confusión (en `runs/detect/val/`).

### Resultados

| Clase | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| Taza | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Botella | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Tijeras | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Destornillador | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Celular | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| **Total** | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

Matriz de confusión: `[COMPLETAR: ruta de la imagen, ej. metricas/confusion_matrix.png]`

El detalle completo está en `metricas.md`.

---

## 9. Rendimiento

| Equipo | CPU               | Resolución | `IMGSZ` | FPS medidos |
|--------|------------------|-------  --|------------|
| laptop | intel i3 8va  gen|   640×480 | 256        |   15   |

El programa muestra el FPS real en pantalla y lo marca en rojo si baja de 15.

---

## 10. Limitaciones conocidas

- **Solo Windows:** la cámara se abre con `cv2.CAP_DSHOW`, que es exclusivo de Windows. En Linux o macOS hay que quitar ese parámetro (`cv2.VideoCapture(0)`).
- **Resolución de inferencia baja (256 px):** se sacrifica precisión en objetos pequeños, lejanos o delgados (por ejemplo, el destornillador) a cambio de FPS.
- **Sensibilidad a la iluminación y al fondo:** con poca luz o fondos muy cargados aumentan los falsos positivos y las detecciones perdidas.
- **Oclusión parcial:** los objetos muy tapados pueden no detectarse o perder su ID de rastreo.
- **Objetos desconocidos:** se ignoran, pero no se etiquetan explícitamente como `DESCONOCIDO` en pantalla.
- **Dependencia de la cámara:** si la cámara deja de responder más de 60 lecturas seguidas, el programa se cierra.

## 11. Mejoras propuestas

- Persistencia en varios frames (confirmar una detección solo tras N frames consecutivos).
- Umbrales de confianza distintos por clase.
- Filtro por área y relación de aspecto de la caja.
- Mostrar explícitamente `DESCONOCIDO` para objetos ajenos.
- Exportar a ONNX o NCNN para acelerar en CPU y correr en dispositivos embebidos (por ejemplo, Raspberry Pi).
- Ampliar el dataset con más fondos, distancias e iluminaciones.
- Hacer la fuente de video configurable por línea de comandos (`--camara 1`, `--url`).
- Compatibilidad multiplataforma de la captura.
- Optimizacion para correr en un sistema raspberry pi o parecido 

---

## 12. Entregables del reto

- [x] Código fuente (`main.py`)
- [X] `requirements.txt`
- [x] `README.md`
- [ ] `respuestas.md`
- [ ] Carpeta `dataset/` (≥ 100 imágenes, 20 por clase, con etiquetas y conjunto de prueba identificado)
- [X] Modelo (`best.pt`)
- [ ] Reporte de métricas (`metricas.md` o `.csv`) con script y pasos
- [ ] Video de demostración (máx. 2 minutos)
