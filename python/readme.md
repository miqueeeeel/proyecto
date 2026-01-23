# ♟️ Chess Analyzer with Stockfish (Python)

Aplicación en Python para analizar posiciones de ajedrez utilizando el motor **Stockfish**.  
Permite obtener el mejor movimiento y la evaluación de una posición, similar al análisis post-partida de plataformas como Chess.com, pero **en local y gratis**.

---

## 🧠 ¿Qué hace esta aplicación?

Este proyecto utiliza:
- **Python** como lenguaje principal
- **Stockfish** como motor de ajedrez
- **python-chess** para manejar el tablero y los movimientos

La aplicación:
1. Crea un tablero de ajedrez
2. Lanza Stockfish como proceso externo
3. Envía la posición actual al motor
4. Recibe el mejor movimiento y la evaluación
5. Muestra el resultado por consola

> Python no calcula los movimientos, solo se comunica con Stockfish.

---

## 🎯 ¿Para qué sirve?

- Analizar partidas propias
- Aprender cómo funcionan los motores de ajedrez
- Practicar integración de librerías externas
- Base para futuros proyectos:
  - Entrenador de ajedrez
  - Juego contra la máquina
  - Backend REST
  - Análisis desde imágenes o FEN

---

## 🧩 Requisitos

### 🔹 Sistema
- Linux (Ubuntu / Debian recomendado)
- Python **3.10 o superior**

Comprobar versión:
```bash
python3 --version
```

--- 

## Motor de ajedrez

```
sudo apt install stockfish
```
comprobar
```
stockfish
```
## dependencias Python
se recomienda usar un entorno virtual.
```
python3 -m venv venv
source venv/bin/activate
```
instalar dependencias
```
python-chess
```
## 📌 Notas

No tiene interfaz gráfica (solo consola) por ahora 

Ideal para ejecutar análisis post-partida

Pensado como base para futuras ampliaciones

## 📄 Licencia
Proyecto personal con fines educativos.

---

## 🚧 Futuras mejoras

Este proyecto está pensado como una base sobre la que seguir construyendo.  
Algunas ideas de evolución son:

### ♟️ Análisis avanzado
- Análisis automático de partidas completas (PGN)
- Comparación entre el movimiento jugado y el mejor movimiento
- Detección de blunders, errores e imprecisiones
- Gráfica de evaluación a lo largo de la partida

### 🖼️ Entrada desde imagen
- Detección del tablero a partir de una imagen (OpenCV)
- Reconocimiento de piezas y generación automática de FEN
- Análisis instantáneo de la posición detectada

### 🎮 Interfaz de usuario
- Interfaz gráfica en escritorio (Tkinter / PyQt)
- Interfaz web con backend en Python o NestJS
- Visualización del tablero y sugerencias en tiempo real

### 🌐 API / Backend
- Exponer el análisis como API REST
- Endpoint para evaluar posiciones FEN
- Integración con frontend Angular
- Análisis asincrónico de partidas

### 🤖 Modo entrenador
- Sugerencias de jugadas con explicación
- Nivel de dificultad configurable
- Análisis enfocado a principiantes o intermedios

### ⚙️ Optimización
- Configuración de profundidad y tiempo de análisis
- Uso de múltiples hilos de Stockfish
- Cacheo de posiciones ya analizadas

---

