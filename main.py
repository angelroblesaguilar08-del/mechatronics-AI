
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

torch.set_num_threads(4)

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELO = str(BASE_DIR / "best.pt")

IMGSZ = 256
CAM_W, CAM_H = 640, 480
FPS_MIN_SLIDER = 5

CONOCIDOS = {
    "bottle": "Botella",
    "cup": "Taza",
    "scissors": "Tijeras",
    "cell phone": "Celular",
    "screwdriver": "Destornillador",
}

COLOR_CAJA = (0, 255, 0)  # verde (BGR)
VENTANA = "Deteccion y Rastreo YOLO - Hack Knights"


def nada(val):
    """Funcion requerida por OpenCV para las barras deslizantes."""
    pass


def dibujar_panel_hud(frame, fps_actual, conteo_objetos):
    """Panel con FPS y conteo por objeto."""
    alto_f, ancho_f = frame.shape[:2]
    x1, y1 = 10, 10
    x2 = min(250, ancho_f - 1)
    y2 = min(95 + 22 * len(conteo_objetos), alto_f - 1)

    # Oscurecer solo el panel
    frame[y1:y2, x1:x2] = cv2.convertScaleAbs(
        frame[y1:y2, x1:x2], alpha=0.4
    )

    cv2.putText(
        frame, "HACK KNIGHTS AI", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )

    color_fps = (0, 255, 0) if fps_actual >= 15 else (0, 0, 255)
    cv2.putText(
        frame, f"FPS Real: {int(fps_actual)}", (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_fps, 2
    )
    cv2.line(frame, (20, 65), (240, 65), (100, 100, 100), 1)

    y_pos = 85
    for nombre_es, cantidad in conteo_objetos.items():
        color_texto = (0, 255, 0) if cantidad > 0 else (180, 180, 180)
        cv2.putText(
            frame, f"{nombre_es}: {cantidad}", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1
        )
        y_pos += 22


def main():
    print("1. Cargando modelo YOLO...")

    # Comprobar que el archivo del modelo exista
    if not Path(MODELO).is_file():
        print(f"ERROR: No se encontro el modelo: {MODELO}")
        print("Coloca el archivo best.pt junto a main.py.")
        return

    model = YOLO(MODELO, task="detect")
    print("Modelo cargado correctamente.")

    ids_interes = [
        i for i, n in model.names.items()
        if n in CONOCIDOS
    ]
    nombres_es = {
        i: CONOCIDOS[model.names[i]]
        for i in ids_interes
    }

    # Calentamiento del modelo y del tracker
    model.track(
        np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8),
        verbose=False,
        imgsz=IMGSZ,
        tracker="bytetrack.yaml",
        persist=True
    )

    print("2. Inicializando camara web...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: No se pudo abrir la camara web.")
        return

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Confianza (%)", VENTANA, 50, 100, nada)
    cv2.createTrackbar("Limite FPS", VENTANA, 20, 30, nada)

    t_anterior = time.perf_counter()
    fps_real = 0.0
    fallos = 0

    try:
        while True:
            t_inicio = time.perf_counter()

            # Leer sliders
            try:
                if cv2.getWindowProperty(
                    VENTANA, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

                conf_slider = cv2.getTrackbarPos(
                    "Confianza (%)", VENTANA
                )
                fps_slider = cv2.getTrackbarPos(
                    "Limite FPS", VENTANA
                )
            except cv2.error:
                break

            conf_min = min(0.95, max(0.10, conf_slider / 100.0))
            fps_limite = max(FPS_MIN_SLIDER, fps_slider)

            # Capturar frame
            ret, frame = cap.read()

            if not ret or frame is None:
                fallos += 1

                if fallos > 60:
                    print("ERROR: La camara dejo de responder.")
                    break

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

                continue

            fallos = 0

            # Inferencia y seguimiento ByteTrack
            r = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=conf_min,
                classes=ids_interes,
                imgsz=IMGSZ
            )[0]

            conteo = {v: 0 for v in CONOCIDOS.values()}
            b = r.boxes

            if len(b) > 0:
                xyxy = b.xyxy.cpu().numpy().astype(int)
                clases = b.cls.cpu().numpy().astype(int)
                confs = b.conf.cpu().numpy() * 100

                # Obtener IDs asignados por ByteTrack
                track_ids = (
                    b.id.cpu().numpy().astype(int)
                    if b.id is not None
                    else [None] * len(b)
                )

                for (x1, y1, x2, y2), c, p, tid in zip(
                    xyxy, clases, confs, track_ids
                ):
                    nombre_es = nombres_es[c]
                    conteo[nombre_es] += 1

                    # Etiqueta con ID del rastreador
                    if tid is not None:
                        etiqueta = f"#{tid} {nombre_es}: {p:.1f}%"
                    else:
                        etiqueta = f"{nombre_es}: {p:.1f}%"

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), COLOR_CAJA, 2
                    )
                    cv2.putText(
                        frame,
                        etiqueta,
                        (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLOR_CAJA,
                        2
                    )

            # Calcular FPS reales
            t_actual = time.perf_counter()
            dt = t_actual - t_anterior
            t_anterior = t_actual

            if dt > 0:
                inst = 1.0 / dt
                fps_real = (
                    inst if fps_real == 0
                    else 0.9 * fps_real + 0.1 * inst
                )

            dibujar_panel_hud(frame, fps_real, conteo)
            cv2.imshow(VENTANA, frame)

            # Limite de FPS
            restante = (
                1.0 / fps_limite
                - (time.perf_counter() - t_inicio)
            )
            espera_ms = max(1, int(restante * 1000))

            if cv2.waitKey(espera_ms) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Programa finalizado correctamente.")


if __name__ == "__main__":
    main()