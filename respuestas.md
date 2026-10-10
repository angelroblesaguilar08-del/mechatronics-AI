# Respuestas técnicas — Equipo [NOMBRE]

> IMPORTANTE: revisa el documento oficial del reto. Si trae preguntas numeradas, usa
> ESAS preguntas como encabezados. Abajo va una plantilla con las preguntas típicas.

## 1. ¿Por qué eligieron YOLO (y la versión n/nano)?
- Detector de una sola etapa: una pasada por imagen, ideal para tiempo real.
- La versión nano es la más ligera: necesario para ≥15 FPS en CPU sin GPU.
- Compromiso: menos precisión a cambio de velocidad. [Explica cómo lo midieron]

## 2. ¿Cómo está compuesto su dataset?
- Total de imágenes: [N]; por clase: cup [ ], bottle [ ], scissors [ ], screwdriver [ ], cell phone [ ].
- Cómo se capturaron (script `capurar_dataset.py`, cámara usada, variaciones de luz/fondo/posición): [ ]
- Etiquetado: [Roboflow, una caja por objeto, criterio con objetos parciales]
- Partición train/val/test: [70/20/10] y cómo garantizaron que test no se usó para entrenar.

## 3. ¿Qué técnicas de aumento de datos usaron y por qué?
[Volteo, brillo, rotación, mosaico... y por qué ayudan a generalizar]

## 4. Hiperparámetros y entrenamiento
Épocas [ ], imgsz [ ], batch [ ], optimizador [ ], herramienta [Colab/GPU]. ¿Hubo sobreajuste? [Evidencia: curvas train vs val]

## 5. ¿Cómo manejan los falsos positivos?
Filtro por clase, umbral de confianza ajustable (50% por defecto), ByteTrack. Resultados observados: [ejemplos reales]

## 6. ¿Cómo logran ≥15 FPS en CPU?
imgsz=256, resolución de captura 640×480, modelo nano, tope de FPS. FPS medidos: [ ] en [equipo]. Qué probaron que NO funcionó: [ ]

## 7. ¿Cuál es la clase más difícil y por qué?
[Probablemente destornillador: delgado/pequeño. Respaldar con métricas de metricas.md]

## 8. Limitaciones y mejoras futuras
[Ver README secciones 10 y 11]

## 9. Bonus (ONNX / embebido / seguimiento / contador)
[Resultados de ONNX: FPS .pt vs .onnx. Objetivo Raspberry Pi 4 (4 GB)]

## 10. Uso de herramientas externas e IA
[Declaren honestamente qué usaron: Ultralytics, Roboflow, Colab, asistentes de IA para código, etc.]

## 11. Reparto de trabajo
| Integrante | Responsabilidad |
|---|---|
| [ ] | [ ] |
