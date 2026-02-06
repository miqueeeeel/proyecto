import cv2
import numpy as np
import logging
import chess
import chess.engine
from pathlib import Path

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)  # INFO para output limpio
logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
CARPETA_TEMPLATES = BASE_DIR / "templates"
RUTA_STOCKFISH = BASE_DIR.parent / "stockfish" / "stockfish-ubuntu-x86-64"

ESTILO_FIJO = "neo"
UMBRAL_COINCIDENCIA = 0.45  # Bajado más para piezas blancas
DEBUG_CASILLAS = True

CARPETA_DEBUG = Path("/home/daw/Documents/proyecto/python/debugIMG")
CARPETA_DEBUG.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DETECCIÓN COLORES TABLERO
# --------------------------------------------------
def detectar_colores_tablero(img):
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(
        data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    counts = np.bincount(labels.flatten())
    idx = np.argsort(counts)[-2:]
    return centers[idx[0]].astype(np.uint8), centers[idx[1]].astype(np.uint8)


def mascara_tablero(img, c1, c2, tol=25):
    diff1 = cv2.absdiff(img, c1)
    diff2 = cv2.absdiff(img, c2)
    m1 = np.sum(diff1, axis=2) < tol
    m2 = np.sum(diff2, axis=2) < tol
    return (np.logical_or(m1, m2).astype(np.uint8) * 255)


def detectar_lineas_grid(mask):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=300, maxLineGap=10)

    if lines is None:
        raise ValueError("No se detectaron líneas en el tablero")

    verticales, horizontales = [], []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if abs(x1 - x2) < 10:
            verticales.append(x1)
        if abs(y1 - y2) < 10:
            horizontales.append(y1)

    if len(verticales) < 2 or len(horizontales) < 2:
        raise ValueError("No se detectaron suficientes líneas del tablero")

    verticales = np.linspace(min(verticales), max(verticales), 9).astype(int)
    horizontales = np.linspace(min(horizontales), max(horizontales), 9).astype(int)

    return verticales, horizontales


def obtener_casillas_reales(img):
    c1, c2 = detectar_colores_tablero(img)
    mask = mascara_tablero(img, c1, c2)
    v_lines, h_lines = detectar_lineas_grid(mask)

    logger.info(f"Líneas horizontales (y): {h_lines}")
    logger.info(f"Primera fila será desde y={h_lines[0]} hasta y={h_lines[1]}")
    logger.info(f"Última fila será desde y={h_lines[7]} hasta y={h_lines[8]}")

    casillas = []
    # Leer en el orden que aparece en la imagen (top to bottom)
    for fila in range(8):
        fila_casillas = []
        for col in range(8):
            x1 = v_lines[col]
            x2 = v_lines[col + 1]
            y1 = h_lines[fila]
            y2 = h_lines[fila + 1]
            fila_casillas.append(img[y1:y2, x1:x2])
        casillas.append(fila_casillas)

    logger.debug("64 casillas reales detectadas")
    return casillas


# --------------------------------------------------
# TEMPLATES
# --------------------------------------------------
def cargar_templates():
    """
    Carga templates y los organiza por tipo de pieza (sin color).
    Retorna diccionario con forma -> [templates]
    """
    templates_white = {}
    templates_black = {}
    pieces_dir = CARPETA_TEMPLATES / "pieces" / ESTILO_FIJO

    if not pieces_dir.exists():
        raise FileNotFoundError(f"Directorio de templates no encontrado: {pieces_dir}")

    mapping_white = {
        "wp": "P", "wn": "N", "wb": "B", "wr": "R", "wq": "Q", "wk": "K"
    }
    
    mapping_black = {
        "bp": "p", "bn": "n", "bb": "b", "br": "r", "bq": "q", "bk": "k"
    }

    # Cargar templates blancos
    for archivo in pieces_dir.iterdir():
        if archivo.suffix.lower() != ".png":
            continue
        stem = archivo.stem.lower()
        
        for code, fen in mapping_white.items():
            if stem.startswith(code):
                img = cv2.imread(str(archivo), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates_white.setdefault(fen, []).append(img)
        
        for code, fen in mapping_black.items():
            if stem.startswith(code):
                img = cv2.imread(str(archivo), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates_black.setdefault(fen, []).append(img)

    if not templates_white or not templates_black:
        raise ValueError("No se cargaron templates de piezas")

    logger.info(f"Templates blancos: {list(templates_white.keys())}")
    logger.info(f"Templates negros: {list(templates_black.keys())}")
    
    return templates_white, templates_black


# --------------------------------------------------
# DETECCIÓN DE COLOR DE PIEZA
# --------------------------------------------------
def detectar_color_pieza(casilla):
    """
    Detecta si una pieza es blanca o negra basándose en los píxeles.
    Retorna: 'white', 'black', o None si no hay pieza
    """
    # Normalizar casilla (eliminar destacados amarillos)
    casilla_norm = normalizar_casilla(casilla)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(casilla_norm, cv2.COLOR_BGR2GRAY)
    
    # Obtener el color del fondo (esquinas de la casilla)
    h, w = gray.shape
    esquinas = np.concatenate([
        gray[0:h//4, 0:w//4].flatten(),
        gray[0:h//4, 3*w//4:w].flatten(),
        gray[3*h//4:h, 0:w//4].flatten(),
        gray[3*h//4:h, 3*w//4:w].flatten()
    ])
    fondo_mean = np.median(esquinas)
    
    # Encontrar píxeles de la pieza (muy diferentes del fondo)
    diff_from_bg = np.abs(gray.astype(float) - fondo_mean)
    pieza_mask = diff_from_bg > 30  # Píxeles que son parte de la pieza
    
    # Si hay muy pocos píxeles de pieza, es casilla vacía
    if np.sum(pieza_mask) < gray.size * 0.08:
        return None
    
    # Extraer solo los píxeles de la pieza
    pieza_pixels = gray[pieza_mask]
    
    # Calcular brillo promedio de la pieza
    pieza_mean = np.mean(pieza_pixels)
    
    # Análisis del área central (más confiable)
    centro = gray[h//3:2*h//3, w//3:2*w//3]
    centro_mask = diff_from_bg[h//3:2*h//3, w//3:2*w//3] > 30
    
    if np.sum(centro_mask) > 0:
        centro_pixels = centro[centro_mask]
        centro_mean = np.mean(centro_pixels)
    else:
        centro_mean = pieza_mean
    
    # Decisión basada en múltiples factores
    # Las piezas blancas tienen relleno muy claro (>180)
    # Las piezas negras/grises tienen relleno más oscuro (<140)
    
    if centro_mean > 165:  # Blanco claro
        return 'white'
    elif centro_mean < 120:  # Gris oscuro/negro
        return 'black'
    else:
        # Zona ambigua - usar píxeles totales
        if pieza_mean > 140:
            return 'white'
        else:
            return 'black'


def normalizar_casilla(casilla):
    """
    Normaliza una casilla para mejorar detección:
    - Elimina casillas destacadas (amarillas)
    """
    hsv = cv2.cvtColor(casilla, cv2.COLOR_BGR2HSV)
    
    # Rango para amarillo (casillas destacadas)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Si hay mucho amarillo, reemplazar
    if np.sum(mask_yellow) > (casilla.shape[0] * casilla.shape[1] * 0.3 * 255):
        casilla_normalizada = casilla.copy()
        # Reemplazar con verde tablero
        casilla_normalizada[mask_yellow > 0] = [120, 150, 100]
    else:
        casilla_normalizada = casilla
    
    return casilla_normalizada


# --------------------------------------------------
# DETECCIÓN PIEZAS
# --------------------------------------------------
def es_casilla_vacia(casilla):
    """Detecta si una casilla está vacía."""
    casilla_norm = normalizar_casilla(casilla)
    gray = cv2.cvtColor(casilla_norm, cv2.COLOR_BGR2GRAY)
    
    std = np.std(gray)
    mean_val = np.mean(gray)
    
    # Buscar píxeles muy diferentes del promedio
    diff_pixels = np.sum(np.abs(gray - mean_val) > 30)
    total_pixels = gray.shape[0] * gray.shape[1]
    diff_ratio = diff_pixels / total_pixels
    
    # Vacía si: baja variación Y pocos píxeles diferentes
    return std < 18 and diff_ratio < 0.15


def identificar_tipo_pieza(casilla, templates, es_blanca=True):
    """
    Identifica el TIPO de pieza usando template matching.
    Usa diferentes estrategias para piezas blancas vs negras.
    """
    casilla_norm = normalizar_casilla(casilla)
    gray = cv2.cvtColor(casilla_norm, cv2.COLOR_BGR2GRAY)
    
    # Preprocesamiento básico
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    ch, cw = gray.shape[:2]
    mejor_score = -1
    mejor_tipo = None
    
    # Para piezas BLANCAS: usar detección de bordes (tienen contorno negro)
    if es_blanca:
        edges = cv2.Canny(gray, 50, 150)
        gray_proc = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
        gray_proc = cv2.normalize(gray_proc, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # Para piezas NEGRAS: usar solo escala de grises normalizada
        # (no tienen bordes fuertes, son grises sólidos)
        gray_proc = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # Opcional: invertir para mejor match
        # gray_proc = 255 - gray_proc

    for pieza_letra, imgs in templates.items():
        tipo = pieza_letra.upper()
        max_score_tipo = 0
        
        for tpl in imgs:
            # Preprocesar template igual que la casilla
            tpl_proc = cv2.GaussianBlur(tpl, (3, 3), 0)
            
            if es_blanca:
                edges_tpl = cv2.Canny(tpl_proc, 50, 150)
                tpl_proc = cv2.addWeighted(tpl_proc, 0.7, edges_tpl, 0.3, 0)
                tpl_proc = cv2.normalize(tpl_proc, None, 0, 255, cv2.NORM_MINMAX)
            else:
                tpl_proc = cv2.normalize(tpl_proc, None, 0, 255, cv2.NORM_MINMAX)
            
            th, tw = tpl_proc.shape[:2]
            
            # Probar varias escalas
            escalas = [0.6, 0.75, 0.85, 0.95, 1.0, 1.1, 1.25, 1.4] if not es_blanca else [0.6, 0.75, 0.9, 1.0, 1.1, 1.25]
            
            for scale_factor in escalas:
                scale = min(ch / th, cw / tw) * scale_factor
                new_w, new_h = int(tw * scale), int(th * scale)
                
                if new_w < 10 or new_h < 10 or new_w > cw or new_h > ch:
                    continue
                    
                tpl_r = cv2.resize(tpl_proc, (new_w, new_h))

                # Probar diferentes métodos de matching
                for method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
                    res = cv2.matchTemplate(gray_proc, tpl_r, method)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    
                    max_score_tipo = max(max_score_tipo, score)

                    if score > mejor_score:
                        mejor_score = score
                        mejor_tipo = tipo
        
        # Log para debugging
        if max_score_tipo > 0.3:
            logger.debug(f"  {tipo}: {max_score_tipo:.3f}")

    logger.debug(f"Mejor match: {mejor_tipo} con score {mejor_score:.3f}")
    
    # Umbral más bajo para piezas negras
    umbral = UMBRAL_COINCIDENCIA if es_blanca else UMBRAL_COINCIDENCIA * 0.85
    
    if mejor_score >= umbral:
        return mejor_tipo

    return None


def identificar_pieza(casilla, templates_white, templates_black):
    """
    Identifica la pieza completa (tipo + color).
    """
    if es_casilla_vacia(casilla):
        return None
    
    # 1. Detectar color
    color = detectar_color_pieza(casilla)
    
    if color is None:
        logger.debug(f"Color detectado: None (casilla vacía)")
        return None
    
    logger.debug(f"Color detectado: {color}")
    
    # 2. Detectar tipo usando templates apropiados
    if color == 'white':
        tipo = identificar_tipo_pieza(casilla, templates_white, es_blanca=True)
        if tipo:
            logger.debug(f"Tipo detectado: {tipo} -> {tipo.upper()}")
            return tipo.upper()  # P, N, B, R, Q, K
        else:
            logger.debug(f"No se detectó tipo para pieza blanca")
    else:  # black
        tipo = identificar_tipo_pieza(casilla, templates_black, es_blanca=False)
        if tipo:
            logger.debug(f"Tipo detectado: {tipo} -> {tipo.lower()}")
            return tipo.lower()  # p, n, b, r, q, k
        else:
            logger.debug(f"No se detectó tipo para pieza negra")
    
    return None


# --------------------------------------------------
# DETERMINAR DERECHOS DE ENROQUE
# --------------------------------------------------
def determinar_castling_rights(board_array):
    """Determina los derechos de enroque."""
    castling = ""
    
    # Enroque blanco (fila 7 = fila 1 del tablero)
    if board_array[7][4] == 'K':
        if board_array[7][7] == 'R':
            castling += 'K'
        if board_array[7][0] == 'R':
            castling += 'Q'
    
    # Enroque negro (fila 0 = fila 8 del tablero)
    if board_array[0][4] == 'k':
        if board_array[0][7] == 'r':
            castling += 'k'
        if board_array[0][0] == 'r':
            castling += 'q'
    
    return castling if castling else '-'


def mostrar_tablero_visual(board_array):
    """
    Muestra el tablero de forma visual en el terminal.
    """
    print("\n" + "="*50)
    print("TABLERO DETECTADO:")
    print("="*50)
    
    # Símbolos Unicode para las piezas
    simbolos = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
        None: '·'
    }
    
    print("  +---+---+---+---+---+---+---+---+")
    for fila_idx in range(8):
        fila_num = 8 - fila_idx
        fila = board_array[fila_idx]
        
        # Imprimir fila con piezas
        print(f"{fila_num} |", end="")
        for pieza in fila:
            simbolo = simbolos.get(pieza, '·')
            print(f" {simbolo} |", end="")
        print()
        print("  +---+---+---+---+---+---+---+---+")
    
    print("    a   b   c   d   e   f   g   h")
    print()


# --------------------------------------------------
# IMAGEN -> FEN
# --------------------------------------------------
def imagen_a_fen(ruta):
    img = cv2.imread(str(ruta))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")
        
    casillas = obtener_casillas_reales(img)
    templates_white, templates_black = cargar_templates()

    board_array = []
    filas = []

    for fila in range(8):
        fen_fila = ""
        vacios = 0
        fila_piezas = []

        for col in range(8):
            casilla = casillas[fila][col]

            if DEBUG_CASILLAS:
                ruta_debug = CARPETA_DEBUG / f"debug_{fila}_{col}.png"
                cv2.imwrite(str(ruta_debug), casilla)

            pieza = identificar_pieza(casilla, templates_white, templates_black)

            if pieza is None:
                vacios += 1
                fila_piezas.append(None)
            else:
                if vacios:
                    fen_fila += str(vacios)
                    vacios = 0
                fen_fila += pieza
                fila_piezas.append(pieza)

        if vacios:
            fen_fila += str(vacios)

        filas.append(fen_fila)
        board_array.append(fila_piezas)

    # Las filas están en orden de imagen (top to bottom = fila 8 a fila 1)
    # FEN requiere fila 8 primero, así que NO invertir
    
    # Mostrar tablero visual
    mostrar_tablero_visual(board_array)
    
    # Determinar derechos de enroque
    castling_rights = determinar_castling_rights(board_array)

    # Construir FEN - las filas ya están en orden correcto
    fen = "/".join(filas) + f" w {castling_rights} - 0 1"
    
    logger.info(f"Primeras 3 filas FEN: {filas[0]} / {filas[1]} / {filas[2]}")
    
    logger.info(f"FEN generado: {fen}")
    
    # Validar FEN
    try:
        board = chess.Board(fen)
        
        # Verificar reyes
        white_kings = sum(1 for row in board_array for piece in row if piece == 'K')
        black_kings = sum(1 for row in board_array for piece in row if piece == 'k')
        
        if white_kings != 1 or black_kings != 1:
            error_msg = f"Posición ilegal: {white_kings} rey(es) blanco(s), {black_kings} rey(es) negro(s)"
            logger.error(error_msg)
            logger.warning(f"Umbral actual: {UMBRAL_COINCIDENCIA}")
            
            logger.info("Tablero detectado:")
            for i, row in enumerate(board_array):
                logger.info(f"Fila {8-i}: {['.' if p is None else p for p in row]}")
            
            raise ValueError(error_msg)
            
    except ValueError as e:
        logger.error(f"FEN inválido generado: {fen}")
        logger.error(f"Error: {e}")
        raise
    
    return fen


# --------------------------------------------------
# STOCKFISH
# --------------------------------------------------
def analizar(fen):
    """Analiza una posición FEN con Stockfish."""
    try:
        board = chess.Board(fen)
    except ValueError as e:
        logger.error(f"FEN inválido: {fen}")
        raise ValueError(f"No se puede analizar FEN inválido: {e}")
    
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(RUTA_STOCKFISH))
        info = engine.analyse(board, chess.engine.Limit(depth=12))
        
        move = info["pv"][0]
        score = info["score"].white().score(mate_score=10000)
        
        return move, score / 100
    
    except chess.engine.EngineTerminatedError as e:
        logger.error(f"Stockfish se cerró inesperadamente: {e}")
        raise
    
    finally:
        if engine:
            try:
                engine.quit()
            except chess.engine.EngineTerminatedError:
                pass


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    imagen_path = CARPETA_TEMPLATES / "tablero" / "imatge.png"

    try:
        fen = imagen_a_fen(imagen_path)
        move, score = analizar(fen)

        print("="*60)
        print(f"FEN: {fen}")
        print(f"Mejor movimiento: {move}")
        print(f"Evaluación: {score:+.2f}")
        print("="*60)
        
    except Exception as e:
        logger.exception("Error en el análisis:")
        print(f"\n✗ Error: {e}")