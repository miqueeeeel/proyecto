import cv2
import numpy as np
import logging
import chess
import chess.engine
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
RUTA_STOCKFISH = BASE_DIR.parent / "stockfish" / "stockfish-ubuntu-x86-64"

ESTILO_FIJO = "neo"
DEBUG_CASILLAS = True  # Cambiar a True para debug

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

    casillas = []
    for fila in range(8):
        fila_casillas = []
        for col in range(8):
            x1 = v_lines[col]
            x2 = v_lines[col + 1]
            y1 = h_lines[fila]
            y2 = h_lines[fila + 1]
            fila_casillas.append(img[y1:y2, x1:x2])
        casillas.append(fila_casillas)

    return casillas


# --------------------------------------------------
# TEMPLATES
# --------------------------------------------------
def cargar_templates():
    """
    Carga templates separados por color.
    Retorna: (templates_white, templates_black)
    donde cada uno es un dict: {'P': [img1, img2], 'N': [...], ...}
    """
    templates_white = {}
    templates_black = {}
    pieces_dir = CARPETA_TEMPLATES / "pieces" / ESTILO_FIJO

    if not pieces_dir.exists():
        raise FileNotFoundError(f"Directorio de templates no encontrado: {pieces_dir}")

    # Mapeo: prefijo archivo → letra FEN
    mapping = {
        'wp': 'P', 'wn': 'N', 'wb': 'B', 'wr': 'R', 'wq': 'Q', 'wk': 'K',
        'bp': 'p', 'bn': 'n', 'bb': 'b', 'br': 'r', 'bq': 'q', 'bk': 'k'
    }

    for archivo in pieces_dir.iterdir():
        if archivo.suffix.lower() != ".png":
            continue
        
        stem = archivo.stem.lower()
        img = cv2.imread(str(archivo), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        # Determinar qué pieza es
        for prefix, fen_char in mapping.items():
            if stem.startswith(prefix):
                if prefix.startswith('w'):
                    templates_white.setdefault(fen_char, []).append(img)
                else:
                    templates_black.setdefault(fen_char, []).append(img)
                break

    if not templates_white or not templates_black:
        raise ValueError("No se cargaron templates de piezas")

    logger.info(f"Templates blancos: {list(templates_white.keys())}")
    logger.info(f"Templates negros: {list(templates_black.keys())}")
    
    return templates_white, templates_black


# --------------------------------------------------
# NORMALIZACIÓN
# --------------------------------------------------
def normalizar_casilla(casilla):
    """Elimina highlights (amarillo/verde)."""
    hsv = cv2.cvtColor(casilla, cv2.COLOR_BGR2HSV)
    
    # Amarillo
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Verde
    lower_green = np.array([40, 50, 100])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    mask_highlight = cv2.bitwise_or(mask_yellow, mask_green)
    
    if np.sum(mask_highlight) > (casilla.shape[0] * casilla.shape[1] * 0.25 * 255):
        casilla_normalizada = casilla.copy()
        casilla_normalizada[mask_highlight > 0] = [110, 145, 95]
    else:
        casilla_normalizada = casilla
    
    return casilla_normalizada


# --------------------------------------------------
# DETECCIÓN DE CASILLA VACÍA
# --------------------------------------------------
def es_casilla_vacia(casilla):
    """
    Detecta si una casilla está vacía.
    Más estricto para evitar falsos positivos.
    """
    casilla_norm = normalizar_casilla(casilla)
    gray = cv2.cvtColor(casilla_norm, cv2.COLOR_BGR2GRAY)
    
    # Criterio 1: Variación baja
    std = np.std(gray)
    mean_val = np.mean(gray)
    
    # Criterio 2: Pocos píxeles diferentes del promedio
    diff_pixels = np.sum(np.abs(gray - mean_val) > 35)  # Aumentado de 30
    total_pixels = gray.shape[0] * gray.shape[1]
    diff_ratio = diff_pixels / total_pixels
    
    # Criterio 3: Analizar el centro específicamente
    h, w = gray.shape
    centro = gray[h//3:2*h//3, w//3:2*w//3]
    std_centro = np.std(centro)
    
    # Vacía si: baja variación global Y baja variación en centro Y pocos píxeles diferentes
    es_vacia = std < 20 and std_centro < 15 and diff_ratio < 0.12
    
    return es_vacia


# --------------------------------------------------
# TEMPLATE MATCHING SIMPLE
# --------------------------------------------------
def match_template(casilla_gray, template, usar_bordes=False):
    """
    Hace template matching entre una casilla y un template.
    Retorna el mejor score encontrado.
    """
    ch, cw = casilla_gray.shape[:2]
    th, tw = template.shape[:2]
    
    mejor_score = -1
    
    # Preprocesar casilla
    casilla_proc = cv2.GaussianBlur(casilla_gray, (3, 3), 0)
    
    if usar_bordes:
        # Para piezas blancas: usar bordes
        edges = cv2.Canny(casilla_proc, 50, 150)
        casilla_proc = cv2.addWeighted(casilla_proc, 0.7, edges, 0.3, 0)
    
    casilla_proc = cv2.normalize(casilla_proc, None, 0, 255, cv2.NORM_MINMAX)
    
    # Preprocesar template igual
    template_proc = cv2.GaussianBlur(template, (3, 3), 0)
    
    if usar_bordes:
        edges_tpl = cv2.Canny(template_proc, 50, 150)
        template_proc = cv2.addWeighted(template_proc, 0.7, edges_tpl, 0.3, 0)
    
    template_proc = cv2.normalize(template_proc, None, 0, 255, cv2.NORM_MINMAX)
    
    # Probar múltiples escalas
    escalas = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    for scale_factor in escalas:
        scale = min(ch / th, cw / tw) * scale_factor
        new_w, new_h = int(tw * scale), int(th * scale)
        
        if new_w < 10 or new_h < 10 or new_w > cw or new_h > ch:
            continue
            
        tpl_resized = cv2.resize(template_proc, (new_w, new_h))
        
        # Probar dos métodos
        for method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
            try:
                res = cv2.matchTemplate(casilla_proc, tpl_resized, method)
                _, score, _, _ = cv2.minMaxLoc(res)
                mejor_score = max(mejor_score, score)
            except:
                continue
    
    return mejor_score


def identificar_pieza(casilla, templates_white, templates_black):
    """
    Identifica la pieza con un enfoque híbrido:
    1. Análisis rápido para determinar si es blanca/negra/vacía
    2. Template matching solo contra candidatos relevantes
    """
    if es_casilla_vacia(casilla):
        return None
    
    casilla_norm = normalizar_casilla(casilla)
    gray = cv2.cvtColor(casilla_norm, cv2.COLOR_BGR2GRAY)
    
    # PASO 1: Análisis rápido para determinar color probable
    h, w = gray.shape
    
    # Detectar fondo (esquinas)
    esquinas = np.concatenate([
        gray[0:h//4, 0:w//4].flatten(),
        gray[0:h//4, 3*w//4:w].flatten(),
        gray[3*h//4:h, 0:w//4].flatten(),
        gray[3*h//4:h, 3*w//4:w].flatten()
    ])
    fondo_mean = np.median(esquinas)
    
    # Píxeles de la pieza (centro)
    centro = gray[h//3:2*h//3, w//3:2*w//3]
    diff_from_bg = np.abs(centro.astype(float) - fondo_mean)
    pieza_mask = diff_from_bg > 30
    
    if np.sum(pieza_mask) < centro.size * 0.1:
        return None  # Muy poca diferencia = vacía
    
    pieza_pixels = centro[pieza_mask]
    pieza_mean = np.mean(pieza_pixels)
    pieza_median = np.median(pieza_pixels)
    
    # Análisis adicional: percentil 75
    p75 = np.percentile(pieza_pixels, 75)
    
    # Decisión basada en múltiples factores
    # Piezas blancas: mean > 140 OR (mean > 110 AND p75 > 150)
    # Esto captura tanto piezas muy blancas como blancas en casillas oscuras
    es_probable_blanca = (pieza_mean > 140) or (pieza_mean > 110 and p75 > 150)
    
    logger.debug(f"  Análisis: mean={pieza_mean:.1f}, median={pieza_median:.1f}, p75={p75:.1f}")
    
    # PASO 2: Template matching solo contra el color probable
    scores = {}
    
    if es_probable_blanca:
        # Probar solo templates blancos
        for pieza, templates in templates_white.items():
            max_score = 0
            for template in templates:
                score = match_template(gray, template, usar_bordes=True)
                max_score = max(max_score, score)
            scores[pieza] = max_score
    else:
        # Probar solo templates negros
        for pieza, templates in templates_black.items():
            max_score = 0
            for template in templates:
                score = match_template(gray, template, usar_bordes=False)
                max_score = max(max_score, score)
            scores[pieza] = max_score
    
    # Encontrar el mejor
    if not scores:
        return None
    
    mejor_pieza = max(scores, key=scores.get)
    mejor_score = scores[mejor_pieza]
    
    # Umbral mínimo
    if mejor_score < 0.35:
        return None
    
    logger.debug(f"Probable: {'blanca' if es_probable_blanca else 'negra'}, Mejor: {mejor_pieza} ({mejor_score:.3f})")
    
    return mejor_pieza


# --------------------------------------------------
# CORRECCIÓN DE ERRORES
# --------------------------------------------------
def corregir_errores_basicos(board_array, casillas, templates_white, templates_black):
    """
    Corrige errores obvios como múltiples reyes.
    """
    board_corregido = [fila[:] for fila in board_array]
    
    # Contar reyes
    white_kings = [(r, c) for r in range(8) for c in range(8) if board_corregido[r][c] == 'K']
    black_kings = [(r, c) for r in range(8) for c in range(8) if board_corregido[r][c] == 'k']
    
    # Si hay múltiples reyes blancos
    if len(white_kings) > 1:
        logger.warning(f"Múltiples reyes blancos: {white_kings}")
        # Recalcular scores específicos
        candidatos = []
        for r, c in white_kings:
            casilla = casillas[r][c]
            gray = cv2.cvtColor(normalizar_casilla(casilla), cv2.COLOR_BGR2GRAY)
            
            score_rey = max(match_template(gray, t, True) for t in templates_white.get('K', []))
            score_peon = max(match_template(gray, t, True) for t in templates_white.get('P', []))
            
            candidatos.append((r, c, score_rey - score_peon))
        
        # Mantener solo el mejor
        candidatos.sort(key=lambda x: x[2], reverse=True)
        rey_real = (candidatos[0][0], candidatos[0][1])
        
        for r, c in white_kings:
            if (r, c) != rey_real:
                board_corregido[r][c] = 'P'
                logger.info(f"Cambiando [{r},{c}] K → P")
    
    # Mismo proceso para reyes negros
    if len(black_kings) > 1:
        logger.warning(f"Múltiples reyes negros: {black_kings}")
        candidatos = []
        for r, c in black_kings:
            casilla = casillas[r][c]
            gray = cv2.cvtColor(normalizar_casilla(casilla), cv2.COLOR_BGR2GRAY)
            
            score_rey = max(match_template(gray, t, False) for t in templates_black.get('k', []))
            score_peon = max(match_template(gray, t, False) for t in templates_black.get('p', []))
            
            candidatos.append((r, c, score_rey - score_peon))
        
        candidatos.sort(key=lambda x: x[2], reverse=True)
        rey_real = (candidatos[0][0], candidatos[0][1])
        
        for r, c in black_kings:
            if (r, c) != rey_real:
                board_corregido[r][c] = 'p'
                logger.info(f"Cambiando [{r},{c}] k → p")
    
    return board_corregido


# --------------------------------------------------
# DETERMINAR DERECHOS DE ENROQUE
# --------------------------------------------------
def determinar_castling_rights(board_array):
    """Determina los derechos de enroque."""
    castling = ""
    
    if board_array[7][4] == 'K':
        if board_array[7][7] == 'R':
            castling += 'K'
        if board_array[7][0] == 'R':
            castling += 'Q'
    
    if board_array[0][4] == 'k':
        if board_array[0][7] == 'r':
            castling += 'k'
        if board_array[0][0] == 'r':
            castling += 'q'
    
    return castling if castling else '-'


def mostrar_tablero_visual(board_array):
    """Muestra el tablero de forma visual."""
    print("\n" + "="*50)
    print("TABLERO DETECTADO:")
    print("="*50)
    
    simbolos = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
        None: '·'
    }
    
    print("  +---+---+---+---+---+---+---+---+")
    for fila_idx in range(8):
        fila_num = 8 - fila_idx
        fila = board_array[fila_idx]
        
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

    # Primera pasada: detectar todas las piezas
    detecciones_raw = []

    for fila in range(8):
        fila_detecciones = []
        for col in range(8):
            casilla = casillas[fila][col]

            if DEBUG_CASILLAS:
                ruta_debug = CARPETA_DEBUG / f"debug_{fila}_{col}.png"
                cv2.imwrite(str(ruta_debug), casilla)

            pieza = identificar_pieza(casilla, templates_white, templates_black)
            fila_detecciones.append(pieza)
        
        detecciones_raw.append(fila_detecciones)
    
    # Segunda pasada: corrección de errores
    board_array = corregir_errores_basicos(detecciones_raw, casillas, templates_white, templates_black)
    
    # Construir FEN
    filas = []
    for fila_piezas in board_array:
        fen_fila = ""
        vacios = 0
        
        for pieza in fila_piezas:
            if pieza is None:
                vacios += 1
            else:
                if vacios:
                    fen_fila += str(vacios)
                    vacios = 0
                fen_fila += pieza

        if vacios:
            fen_fila += str(vacios)

        filas.append(fen_fila)

    mostrar_tablero_visual(board_array)
    castling_rights = determinar_castling_rights(board_array)
    fen = "/".join(filas) + f" w {castling_rights} - 0 1"
    
    logger.info(f"FEN generado: {fen}")
    
    # Validar
    try:
        board = chess.Board(fen)
        
        white_kings = sum(1 for row in board_array for piece in row if piece == 'K')
        black_kings = sum(1 for row in board_array for piece in row if piece == 'k')
        
        if white_kings != 1 or black_kings != 1:
            error_msg = f"Posición ilegal: {white_kings} rey(es) blanco(s), {black_kings} rey(es) negro(s)"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except ValueError as e:
        logger.error(f"FEN inválido: {fen}")
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
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        
        move = info["pv"][0]
        score = info["score"].white().score(mate_score=10000)
        
        return move, score / 100
    
    except chess.engine.EngineTerminatedError as e:
        logger.error(f"Stockfish se cerró: {e}")
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