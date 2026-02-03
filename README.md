♟️ Chess Vision Tutor

A computer vision chess tutor that reads a real board from an image, reconstructs the position in FEN, and analyzes it with Stockfish.

This project combines OpenCV, template matching, python-chess, and Stockfish to create a chess tutor capable of:

Reading a chessboard from an image

Detecting pieces and squares using templates

Reconstructing the exact board position in FEN

Asking Stockfish for the best move and evaluation

Acting as an automated tutor for real games

🚀 Features

🧠 Board recognition from image (OpenCV)

🧩 Piece detection using template matching

♟️ Automatic FEN reconstruction

🤖 Stockfish integration for analysis

📊 Position evaluation and best move suggestion

🛠️ Designed to work with screenshots or photos of real boards

🖼️ How it works

The program loads an image of a chessboard

The board is divided into 64 squares

Each square is compared with piece templates

A full board state is reconstructed

The position is converted into FEN

Stockfish analyzes the position

The best move and evaluation are returned

📂 Project Structure
chess-vision-tutor/
│
├── images/                # Board and piece templates
├── stockfish/            # Stockfish engine binary
├── main.py               # Main program
├── detector.py           # Image processing & detection logic
├── fen_builder.py        # FEN reconstruction
└── README.md

🧰 Requirements

Python 3.10+

OpenCV

numpy

python-chess

Stockfish engine

Install dependencies:

pip install opencv-python numpy python-chess


Download Stockfish and place the binary inside the stockfish/ folder.

▶️ Usage
python main.py


The program will:

Load the board image

Detect pieces

Print the FEN

Show Stockfish evaluation and best move

🖼️ Templates

The detection system is based on:

Light squares (beige)

Dark squares (green)

White pieces

Black pieces

Templates must be cropped precisely and have consistent size.

🧠 Example Output
FEN: r1bqkbnr/pppp1ppp/2n5/4p3/1b2P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Best move: Nc3
Evaluation: +0.34

🗺️ Roadmap

 Automatic template generation

 Perspective correction for photos

 Real-time camera mode

 GUI interface

 Move tracking between positions

 Support for different board themes

🎯 Goal

The goal of this project is to build a real chess tutor that can understand a board visually and give professional engine feedback, without manual input.

📸 Future Vision

Eventually, this could evolve into:

A mobile app that reads real boards

A training assistant for over-the-board players

A teaching tool for coaches

👨‍💻 Author

Miquel
Software developer passionate about computer vision and chess.

⭐ Contributing

Contributions, ideas, and improvements are welcome!
