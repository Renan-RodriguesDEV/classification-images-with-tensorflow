"""
Inferência com modelo CNN salvo
Suporte: webcam em tempo real ou arquivo de imagem
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

# ─────────────────────────────────────────
# CONFIGURAÇÕES — ajuste se necessário
# ─────────────────────────────────────────

MODEL_PATH = "modelo_cnn.keras"
CLASSES_PATH = "classes.json"
IMG_SIZE = (128, 128)  # deve ser igual ao usado no treino
CONFIANCA_MIN = 0.5  # abaixo disso exibe "Incerto"


# ─────────────────────────────────────────
# CARREGAMENTO
# ─────────────────────────────────────────


def carregar_modelo():
    """Carrega o modelo e as classes salvas."""
    if not Path(MODEL_PATH).exists():
        print(f"[ERRO] Modelo não encontrado: {MODEL_PATH}")
        sys.exit(1)
    if not Path(CLASSES_PATH).exists():
        print(f"[ERRO] Classes não encontradas: {CLASSES_PATH}")
        sys.exit(1)

    modelo = keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH) as f:
        classes = json.load(f)

    print(f"✅ Modelo carregado | Classes: {classes}")
    return modelo, classes


# ─────────────────────────────────────────
# PREDIÇÃO
# ─────────────────────────────────────────


def prever(frame_bgr: np.ndarray, modelo, classes: list) -> tuple[str, float]:
    """
    Recebe um frame BGR (padrão OpenCV), retorna (classe, confiança).
    """
    # OpenCV usa BGR, o modelo espera RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, IMG_SIZE)
    arr = np.expand_dims(img, axis=0).astype("float32")
    # Nota: não normalizamos aqui pois o Rescaling está dentro do modelo

    probs = modelo.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    confianca = float(probs[idx])
    classe = classes[idx]

    return classe, confianca


# ─────────────────────────────────────────
# MODO 1: ARQUIVO DE IMAGEM
# ─────────────────────────────────────────


def prever_arquivo(caminho: str, modelo, classes: list) -> None:
    """Classifica uma imagem estática e exibe o resultado."""
    frame = cv2.imread(caminho)
    if frame is None:
        print(f"[ERRO] Não foi possível abrir: {caminho}")
        return

    classe, confianca = prever(frame, modelo, classes)

    # Desenha resultado na imagem
    label = f"{classe}  {confianca:.0%}"
    cor = (0, 200, 0) if confianca >= CONFIANCA_MIN else (0, 165, 255)
    cv2.putText(
        frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor, 2, cv2.LINE_AA
    )

    print(f"\n🔍 Resultado: {classe}  ({confianca:.2%})")

    cv2.imshow("Resultado — pressione qualquer tecla para fechar", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────
# MODO 2: WEBCAM EM TEMPO REAL
# ─────────────────────────────────────────


def prever_webcam(modelo, classes: list) -> None:
    """
    Abre a webcam e classifica frame a frame.
    Pressione 'q' para sair.
    """
    cap = cv2.VideoCapture(0)  # 0 = webcam padrão

    if not cap.isOpened():
        print("[ERRO] Webcam não encontrada.")
        return

    print("📷 Webcam aberta | Pressione 'q' para sair\n")

    # Roda a predição a cada 5 segundos (abordagem simples)
    INTERVALO_SEGUNDOS = 5
    ultimo_label = ("...", 0.0)  # guarda o último resultado entre frames

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Faz uma predição e aguarda 5 segundos para a próxima
        ultimo_label = prever(frame, modelo, classes)

        classe, confianca = ultimo_label
        label = f"{classe}  {confianca:.0%}"
        cor = (0, 200, 0) if confianca >= CONFIANCA_MIN else (0, 165, 255)

        # Fundo semitransparente para o texto ficar legível
        cv2.rectangle(frame, (10, 10), (400, 50), (0, 0, 0), -1)
        cv2.putText(
            frame, label, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor, 2, cv2.LINE_AA
        )

        cv2.imshow("CNN ao vivo — 'q' para sair", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(INTERVALO_SEGUNDOS)

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Encerrado.")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Inferência CNN — webcam ou arquivo")
    parser.add_argument(
        "--imagem",
        "-i",
        type=str,
        default=None,
        help="Caminho para uma imagem. Ex: --imagem foto.jpg",
    )
    args = parser.parse_args()

    modelo, classes = carregar_modelo()

    if args.imagem:
        prever_arquivo(args.imagem, modelo, classes)
    else:
        prever_webcam(modelo, classes)


if __name__ == "__main__":
    main()
