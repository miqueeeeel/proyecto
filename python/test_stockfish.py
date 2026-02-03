import os
import cv2
import numpy as np
import logging
import shutil
import subprocess
import chess
import chess.engine
import traceback
from pathlib import Path

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIGURACIÓN FIJA (MODO ESTABLE)
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
CARPETA_TEMPLATES = BASE_DIR / "templates"
RUTA_STOCKFISH = BASE_DIR.parent / "stockfish" / "stockfish-ubuntu-x86-64-avx2"

ESTILO_FIJO = "neo"              # 🔒 estilo único
BOARD_FIJO = "green.png"         # 🔒 tablero único
UMBRAL_COINCIDENCIA = 0.55

# --------------------------------------------------
# TABLERO
# --------------------------------------------------
def detectar_tablero(img):
    """Recorta el tablero eliminando márgenes exteriores"""
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.10)
    tablero = img[margin:h-margin, margin:w-margin]
    logger.info(f"Tablero recortado: {tablero.shape}")
    return tablero

# --------------------------------------------------
# CARGA DE TEMPLATES
# --------------------------------------------------
def cargar_templates():
    """Carga SOLO templates pieces/neo"""
    templates = {}
    pieces_dir = CARPETA_TEMPLATES / "pieces" / ESTILO_FIJO

    mapping = {
        "wp": "P", "bp": "p",
        "wn": "N", "bn": "n",
        "wb": "B", "bb": "b",
        "wr": "R", "br": "r",
        "wq": "Q", "bq": "q",
        "wk": "K", "bk": "k",
    }

    for archivo in pieces_dir.iterdir():
        if archivo.suffix.lower() != ".png":
            continue

        stem = archivo.stem.lower()
        pieza = None
        for code, fen in mapping.items():
            if stem.startswith(code):
                pieza = fen
                break

        if pieza is None:
            continue

        img = cv2.imread(str(archivo), cv2.IMREAD_GRAYSCALE)
        templates.setdefault(pieza, []).append(img)

    for k in templates:
        logger.info(f"Template {k}: {len(templates[k])}")

    return templates

# --------------------------------------------------
# CASILLA VACÍA
# --------------------------------------------------
def es_casilla_vacia(casilla):
    gray = cv2.cvtColor(casilla, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size
    std = np.std(gray)
    return edge_ratio < 0.035 and std < 28

# --------------------------------------------------
# IDENTIFICAR PIEZA
# --------------------------------------------------
def identificar_pieza(casilla, templates):
    gray = cv2.cvtColor(casilla, cv2.COLOR_BGR2GRAY)

    mejor_score = -1
    mejor_pieza = None

    for pieza, imgs in templates.items():
        for tpl in imgs:
            for scale in [0.7, 0.9, 1.0, 1.1]:
                h = int(tpl.shape[0] * scale)
                w = int(tpl.shape[1] * scale)
                if h <= 5 or w <= 5:
                    continue
                if h > gray.shape[0] or w > gray.shape[1]:
                    continue

                tpl_r = cv2.resize(tpl, (w, h))
                res = cv2.matchTemplate(gray, tpl_r, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)

                if score > mejor_score:
                    mejor_score = score
                    mejor_pieza = pieza

    if mejor_score < UMBRAL_COINCIDENCIA:
        return None

    return mejor_pieza

# --------------------------------------------------
# IMAGEN → FEN
# --------------------------------------------------
def imagen_a_fen(ruta):
    img = cv2.imread(str(ruta))
    if img is None:
        logger.error("No se pudo leer la imagen")
        return None

    tablero = detectar_tablero(img)
    templates = cargar_templates()

    h, w = tablero.shape[:2]
    dh, dw = h // 8, w // 8

    filas = []

    for fila in range(8):
        fen_fila = ""
        vacios = 0
        for col in range(8):
            casilla = tablero[
                fila*dh:(fila+1)*dh,
                col*dw:(col+1)*dw
            ]

            if es_casilla_vacia(casilla):
                vacios += 1
            else:
                pieza = identificar_pieza(casilla, templates)
                if pieza is None:
                    vacios += 1
                else:
                    if vacios > 0:
                        fen_fila += str(vacios)
                        vacios = 0
                    fen_fila += pieza

        if vacios > 0:
            fen_fila += str(vacios)

        filas.append(fen_fila)

    filas = filas[::-1]
    fen = "/".join(filas) + " w KQkq - 0 1"
    return fen

# --------------------------------------------------
# STOCKFISH / FALLBACK
# --------------------------------------------------
def analizar(fen):
    try:
        board = chess.Board(fen)
    except:
        return None

    if shutil.which("stockfish"):
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        info = engine.analyse(board, chess.engine.Limit(depth=12))
        engine.quit()
        move = info["pv"][0]
        score = info["score"].white().score(mate_score=10000)
        return move, score / 100

    # fallback material
    return next(iter(board.legal_moves)), 0.0

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    try:
        import sys
        if len(sys.argv) >= 2:
            ruta = Path(sys.argv[1])
        else:
            ruta = CARPETA_TEMPLATES / "boards" / BOARD_FIJO

        fen = imagen_a_fen(ruta)
        logger.info(f"FEN: {fen}")

        move, score = analizar(fen)
        logger.info(f"Mejor movimiento: {move} (score {score})")

    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
