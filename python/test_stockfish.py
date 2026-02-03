import cv2
import numpy as np
import logging
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
# CONFIGURACIÓN
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
CARPETA_TEMPLATES = BASE_DIR / "templates"

# ⚠️ IMPORTANTE: usa la versión SIN avx2
RUTA_STOCKFISH = BASE_DIR.parent / "stockfish" / "stockfish-ubuntu-x86-64"

ESTILO_FIJO = "neo"
BOARD_FIJO = "green.png"
UMBRAL_COINCIDENCIA = 0.4  # Bajado para capturar piezas
DEBUG_CASILLAS = False      # True para guardar imágenes de casillas detectadas

# --------------------------------------------------
# TABLERO
# --------------------------------------------------
def detectar_tablero(img):
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.10)
    tablero = img[margin:h-margin, margin:w-margin]
    logger.info(f"Tablero recortado: {tablero.shape}")
    return tablero

# --------------------------------------------------
# CARGA DE TEMPLATES
# --------------------------------------------------
def cargar_templates():
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
        if img is None:
            continue
        templates.setdefault(pieza, []).append(img)

    for k in templates:
        logger.info(f"Template {k}: {len(templates[k])}")

    return templates

# --------------------------------------------------
# Detección de casilla vacía
# --------------------------------------------------
def es_casilla_vacia(casilla):
    gray = cv2.cvtColor(casilla, cv2.COLOR_BGR2GRAY)
    # Desviación estándar baja → probablemente vacía
    return np.std(gray) < 25

# --------------------------------------------------
# IDENTIFICAR PIEZA
# --------------------------------------------------
def identificar_pieza(casilla, templates):
    gray = cv2.cvtColor(casilla, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]

    mejor_score = -1
    mejor_pieza = None

    for pieza, imgs in templates.items():
        for tpl in imgs:
            th, tw = tpl.shape[:2]

            if th == 0 or tw == 0:
                continue
            base_scale = min(ch / th, cw / tw)
            scale_factors = [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]

            for f in scale_factors:
                scale = base_scale * f
                h = max(1, int(th * scale))
                w = max(1, int(tw * scale))
                if h > ch or w > cw:
                    continue

                tpl_r = cv2.resize(tpl, (w, h), interpolation=cv2.INTER_AREA)

                # Matching directo en intensidad
                try:
                    res = cv2.matchTemplate(gray, tpl_r, cv2.TM_CCOEFF_NORMED)
                    _, score, _, _ = cv2.minMaxLoc(res)
                except Exception:
                    score = -1

                # fallback: matching por bordes
                if score < UMBRAL_COINCIDENCIA:
                    edges_cas = cv2.Canny(gray, 50, 150)
                    edges_tpl = cv2.Canny(tpl_r, 50, 150)
                    try:
                        res_e = cv2.matchTemplate(edges_cas, edges_tpl, cv2.TM_CCOEFF_NORMED)
                        _, score_e, _, _ = cv2.minMaxLoc(res_e)
                    except Exception:
                        score_e = -1
                    if score_e > score:
                        score = score_e

                if score > mejor_score:
                    mejor_score = score
                    mejor_pieza = pieza

    logger.info(f"Score máximo en casilla: {mejor_pieza} = {mejor_score:.3f}")
    if mejor_score >= UMBRAL_COINCIDENCIA:
        return mejor_pieza
    return None

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
            casilla = tablero[fila*dh:(fila+1)*dh, col*dw:(col+1)*dw]

            if DEBUG_CASILLAS:
                cv2.imwrite(f"debug_{fila}_{col}.png", casilla)

            if es_casilla_vacia(casilla):
                vacios += 1
                continue

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

    # Validar FEN antes de devolver
    try:
        chess.Board(fen)
    except ValueError as e:
        logger.error(f"FEN inválido generado: {fen} - {e}")
        return None

    logger.info(f"FEN generado: {fen}")
    return fen

# --------------------------------------------------
# ANALIZAR CON STOCKFISH
# --------------------------------------------------
def analizar(fen):
    try:
        board = chess.Board(fen)
    except ValueError as e:
        logger.error(f"FEN inválido: {fen} - {e}")
        return None, None

    if not board.legal_moves.count():
        logger.warning("Posición sin movimientos legales")
        return None, None

    if RUTA_STOCKFISH.exists():
        try:
            logger.info("Usando Stockfish local")
            engine = chess.engine.SimpleEngine.popen_uci(str(RUTA_STOCKFISH))
            info = engine.analyse(board, chess.engine.Limit(depth=12))
            engine.quit()
            move = info["pv"][0]
            score = info["score"].white().score(mate_score=10000)
            return move, score / 100
        except Exception as e:
            logger.error(f"Error con Stockfish: {e}")
            return None, None

    logger.warning("Stockfish no encontrado, usando fallback")
    return next(iter(board.legal_moves)), 0.0

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    try:
        # Test con FEN conocido
        logger.info("=== Test con FEN conocido ===")
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move, score = analizar(fen)
        if move:
            logger.info(f"Mejor movimiento: {move} (score {score})")
        else:
            logger.warning("No se pudo obtener movimiento")

        # Procesar imagen real
        logger.info("\n=== Procesando imagen ===")
        imagen_path = BASE_DIR / "templates" / "tablero" / "imatge.png"
        
        if imagen_path.exists():
            logger.info(f"Procesando: {imagen_path}")
            fen_detectado = imagen_a_fen(str(imagen_path))
            if fen_detectado:
                move, score = analizar(fen_detectado)
                if move:
                    logger.info(f"Mejor movimiento: {move} (score {score})")
                else:
                    logger.warning("No se pudo obtener movimiento del FEN detectado")
        else:
            logger.warning(f"Imagen no encontrada en: {imagen_path}")

    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
