import cv2
import numpy as np
import os
import chess
import chess.engine

# --- CONFIGURACIÓN ---
RUTA_STOCKFISH = "../stockfish/stockfish-ubuntu-x86-64-avx2" # Ajusta tu ruta
CARPETA_TEMPLATES = "templates"
UMBRAL_COINCIDENCIA = 0.85  # Qué tan exacto debe ser (0.8 a 0.9 suele ir bien)

def cargar_templates():
    """Carga las imágenes de referencia de la carpeta templates"""
    templates = {}
    if not os.path.exists(CARPETA_TEMPLATES):
        os.makedirs(CARPETA_TEMPLATES)
        print(f"⚠️ CREA LA CARPETA '{CARPETA_TEMPLATES}' Y METE LAS IMÁGENES DE LAS PIEZAS.")
        return {}

    for archivo in os.listdir(CARPETA_TEMPLATES):
        if archivo.endswith(".png") or archivo.endswith(".jpg"):
            # El nombre del archivo debe empezar con la letra FEN (P, p, k, q...)
            # Ejemplo: "P.png" o "P_verde.png" -> la clave será 'P'
            pieza_fen = archivo[0] 
            img = cv2.imread(os.path.join(CARPETA_TEMPLATES, archivo), 0) # Cargar en escala de grises
            
            if pieza_fen not in templates:
                templates[pieza_fen] = []
            templates[pieza_fen].append(img)
    return templates

def identificar_pieza(img_casilla, templates):
    """Compara una casilla con todos los templates y devuelve la pieza ganadora"""
    img_gray = cv2.cvtColor(img_casilla, cv2.COLOR_BGR2GRAY)
    
    mejor_score = -1
    mejor_pieza = None

    for pieza, lista_imgs in templates.items():
        for plantilla in lista_imgs:
            # Importante: La plantilla debe ser menor o igual a la casilla
            if plantilla.shape[0] > img_gray.shape[0] or plantilla.shape[1] > img_gray.shape[1]:
                continue
                
            res = cv2.matchTemplate(img_gray, plantilla, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > mejor_score:
                mejor_score = max_val
                mejor_pieza = pieza

    # Si la coincidencia es muy baja, asumimos que la casilla está vacía
    if mejor_score < UMBRAL_COINCIDENCIA:
        return None # Vacío
    
    return mejor_pieza

def imagen_a_fen(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None: return None

    # 1. Recorte (Ajustado a tu imagen)
    alto, ancho, _ = img.shape
    margen_izq = 25
    margen_inf = 25
    tablero = img[0:alto-margen_inf, margen_izq:ancho] # Ajuste aproximado
    
    # Cargar templates
    templates = cargar_templates()
    if not templates:
        return None

    h, w, _ = tablero.shape
    step_h = h // 8
    step_w = w // 8

    fen_rows = []

    # Recorrer filas (FEN empieza desde la fila 8 - arriba)
    for fila in range(8):
        fen_row = ""
        vacios = 0
        
        for col in range(8):
            # Recortar celda
            y1, y2 = fila * step_h, (fila + 1) * step_h
            x1, x2 = col * step_w, (col + 1) * step_w
            casilla = tablero[y1:y2, x1:x2]
            
            # Identificar
            pieza = identificar_pieza(casilla, templates)
            
            if pieza is None:
                vacios += 1
            else:
                if vacios > 0:
                    fen_row += str(vacios)
                    vacios = 0
                fen_row += pieza
        
        if vacios > 0:
            fen_row += str(vacios)
        fen_rows.append(fen_row)

    # Unir filas con '/'
    fen_completo = "/".join(fen_rows)
    
    # Añadir meta-info por defecto (Turno blancas, enroques posibles, etc.)
    # Nota: Detectar de quién es el turno es difícil visualmente, asumimos blancas ('w')
    fen_final = f"{fen_completo} w KQkq - 0 1"
    return fen_final

# --- EJECUCIÓN PRINCIPAL ---

fen_generado = imagen_a_fen("imatge.png")

if fen_generado:
    print(f"FEN Detectado: {fen_generado}")
    
    # Iniciar Stockfish
    board = chess.Board(fen_generado)
    print(board) # Dibuja el tablero en texto para verificar visualmente

    try:
        engine = chess.engine.SimpleEngine.popen_uci(RUTA_STOCKFISH)
        info = engine.analyse(board, chess.engine.Limit(depth=18))
        best_move = info["pv"][0]
        score = info["score"].white().score(mate_score=10000)
        
        print("\n" + "="*40)
        print(f"🤖 STOCKFISH DICE: {best_move}")
        print(f"📊 Puntuación: {score/100:.2f}")
        print("="*40)
        engine.quit()
    except Exception as e:
        print(f"Error con Stockfish: {e}")
else:
    print("No se pudo generar el FEN. Revisa la carpeta 'templates'.")