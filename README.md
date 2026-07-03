# 🎮 Paddle Ball — Hand Tracking Game

A webcam-controlled arcade game built with **OpenCV** and **MediaPipe**. Move your index finger in front of your camera to steer a paddle and catch a falling ball. Chain hits together to build combos, rack up score, and beat your high score — all saved locally between sessions.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-orange)

## ✨ Features

- **Hand tracking control** — no keyboard/mouse needed to play, just your index finger
- **Three difficulty modes** — Easy, Normal, and Hard, each with different paddle size, ball speed, and speed ramp-up
- **Combo system** — consecutive catches build a combo multiplier and bonus points, with on-screen combo milestones (x5, x10, x15, x20)
- **Persistent high score** — saved to a local `save.json` file
- **Polished visual effects** — glowing paddle/ball, particle sparks on hits and misses, floating score text, screen shake, and a danger-zone pulse near the paddle line
- **Full game flow** — main menu with difficulty selection, countdown before each round, pause/resume, and a game-over screen with stats

## 🕹️ Controls

| Key | Action |
|-----|--------|
| Move index finger | Move the paddle |
| `1` / `2` / `3` | Select Easy / Normal / Hard difficulty (menu or game-over screen) |
| `SPACE` | Start game / Restart after game over |
| `P` | Pause / Resume |
| `M` | Return to main menu (from game-over screen) |
| `ESC` | Quit |

## 📦 Requirements

- Python 3.9+
- A webcam
- The following Python packages:

```bash
pip install opencv-python mediapipe numpy
```

## 🧩 Model File

This game uses MediaPipe's **HandLandmarker** task model. Download `hand_landmarker.task` and place it in the same directory as the script:
 download it manually from the [MediaPipe Hand Landmarker model page](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) 
## ▶️ Running the Game

```bash
python main.py
```

A window will open showing your webcam feed with the game overlaid on top. Choose a difficulty from the main menu, press `SPACE`, and start catching!

## ⚙️ How It Works

- Each frame is captured from the webcam, flipped horizontally (mirror mode), and passed to MediaPipe's hand landmarker.
- The tip of the index finger (landmark 8) is tracked and smoothed to control the paddle's target position.
- The paddle smoothly interpolates toward the finger position for responsive but fluid movement.
- The ball bounces off the walls and ceiling, and must be caught by the paddle before it falls past the bottom of the screen.
- Successful catches increase score and combo count; combo streaks grant bonus points every 5 hits.
- Missing the ball costs a life. Losing all lives ends the game and updates the high score if beaten.

## 📁 Project Structure

```
.
├── main.py        # Main game script
├── hand_landmarker.task  # MediaPipe hand tracking model (download separately)
└── save.json             # Auto-generated high score save file
```

## 🛠️ Customization

Key constants near the top of the script make it easy to tweak:

- `W`, `H` — window resolution
- `START_LIVES` — number of lives per game
- `FINGER_SMOOTHING`, `PADDLE_SMOOTHING` — control responsiveness/smoothness
- `DIFFICULTIES` — paddle width, ball speed, and speed increase per difficulty tier
