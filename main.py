import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import json
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

W, H = 1280, 720
PADDLE_H = 14
PADDLE_Y = H - 90
BALL_R = 18
START_LIVES = 3
SAVE_FILE = "save.json"

# smoothing
FINGER_SMOOTHING = 0.50
PADDLE_SMOOTHING = 0.95

# colors
COL_PADDLE = (0, 220, 255)
COL_BALL = (255, 0, 255)
COL_TRAIL = (40, 140, 90)
COL_SCORE = (255, 230, 100)
COL_DEAD = (50, 60, 255)
COL_LIVES = (0, 0, 255)
COL_HIGHSCORE = (255, 180, 80)
COL_COMBO = (180, 120, 255)
COL_COMBO_GLOW = (255, 160, 255)
COL_PAUSE = (255, 220, 120)

FONT = cv2.FONT_HERSHEY_DUPLEX

# ---------------- DIFFICULTY PRESETS ----------------
DIFFICULTIES = {
    "easy": {
        "label": "EASY",
        "paddle_w": 210,
        "ball_speed": 14,
        "speed_inc": 0.35,
        "color": (140, 255, 180),
    },
    "normal": {
        "label": "NORMAL",
        "paddle_w": 160,
        "ball_speed": 18,
        "speed_inc": 0.50,
        "color": (255, 230, 120),
    },
    "hard": {
        "label": "HARD",
        "paddle_w": 125,
        "ball_speed": 23,
        "speed_inc": 0.75,
        "color": (100, 140, 255),
    },
}


def lerp(a, b, t):
    return a + (b - a) * t


def draw_glow_circle(img, cx, cy, r, color, layers=4, base_alpha=0.35):
    for i in range(layers, 0, -1):
        radius = r + i * 6
        alpha = base_alpha / i
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_glow_rect(img, x, y, w, h, color, layers=3, base_alpha=0.4):
    r = h // 2
    for i in range(layers, 0, -1):
        ex = x - i * 4
        ey = y - i * 4
        ew = w + i * 8
        eh = h + i * 8
        alpha = base_alpha / i
        overlay = img.copy()
        cv2.rectangle(overlay, (ex + r, ey), (ex + ew - r, ey + eh), color, -1)
        cv2.rectangle(overlay, (ex, ey + r), (ex + ew, ey + eh - r), color, -1)
        cv2.circle(overlay, (ex + r, ey + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (ex + ew - r, ey + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (ex + r, ey + eh - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (ex + ew - r, ey + eh - r), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    cv2.circle(img, (x + r, y + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w - r, y + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + r, y + h - r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w - r, y + h - r), r, color, -1, cv2.LINE_AA)


class Spark:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 9)
        self.x = float(x)
        self.y = float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 4
        self.color = color
        self.life = 1.0
        self.decay = random.uniform(0.04, 0.10)
        self.r = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.35
        self.life -= self.decay
        return self.life > 0

    def draw(self, img):
        alpha = self.life
        c = tuple(int(ch * alpha) for ch in self.color)
        cv2.circle(img, (int(self.x), int(self.y)), self.r, c, -1, cv2.LINE_AA)


class FloatingText:
    def __init__(self, text, x, y, color, life=1.0, vy=-1.2, scale=0.8):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.life = life
        self.vy = vy
        self.scale = scale

    def update(self):
        self.y += self.vy
        self.life -= 0.03
        return self.life > 0

    def draw(self, img):
        alpha = max(0.0, min(1.0, self.life))
        col = tuple(int(c * alpha) for c in self.color)
        cv2.putText(img, self.text, (int(self.x), int(self.y)), FONT, self.scale, (20, 20, 30), 4, cv2.LINE_AA)
        cv2.putText(img, self.text, (int(self.x), int(self.y)), FONT, self.scale, col, 2, cv2.LINE_AA)


class BallCatchGame:
    def __init__(self):
        self.hand_landmarker = self._create_hand_landmarker()
        self.high_score = self.load_high_score()

        self.difficulty = "normal"
        self.selected_difficulty = "normal"   # menu selection
        self._apply_difficulty(self.difficulty)

        self._reset(full_reset=True)

        self.paddle_x = W // 2 - self.paddle_w // 2
        self.paddle_x_tgt = self.paddle_x

        self.finger_x = W // 2
        self.finger_y = H // 2
        self.hand_visible = False

        # menu / game states
        # menu -> countdown -> playing -> paused -> dead
        self.state = "menu"
        self.cam_frame = np.zeros((H, W, 3), dtype=np.uint8)

        self.countdown_start = None
        self.countdown_duration = 3.0
        self.go_duration = 0.65

    # ---------------- HAND LANDMARKER ----------------
    def _create_hand_landmarker(self):
        model_path = "hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        return vision.HandLandmarker.create_from_options(options)

    # ---------------- SAVE / LOAD ----------------
    def load_high_score(self):
        try:
            if os.path.exists(SAVE_FILE):
                with open(SAVE_FILE, "r") as f:
                    data = json.load(f)
                    return int(data.get("high_score", 0))
        except Exception:
            pass
        return 0

    def save_high_score(self):
        try:
            with open(SAVE_FILE, "w") as f:
                json.dump({"high_score": self.high_score}, f)
        except Exception:
            pass

    # ---------------- DIFFICULTY ----------------
    def _apply_difficulty(self, name):
        cfg = DIFFICULTIES[name]
        self.difficulty = name
        self.difficulty_label = cfg["label"]
        self.difficulty_color = cfg["color"]
        self.paddle_w = cfg["paddle_w"]
        self.ball_speed_0 = cfg["ball_speed"]
        self.speed_inc = cfg["speed_inc"]

        if hasattr(self, "paddle_x"):
            self.paddle_x = max(0, min(W - self.paddle_w, self.paddle_x))
            self.paddle_x_tgt = max(0, min(W - self.paddle_w, self.paddle_x_tgt))

    # ---------------- RESET ----------------
    def _reset_ball_only(self):
        self.ball_x = float(W // 2)
        self.ball_y = float(H // 4)

        speed = self.ball_speed_0 + self.speed_inc * self.score
        angle = random.uniform(math.pi * 0.3, math.pi * 0.7)
        self.ball_vx = math.cos(angle) * speed * random.choice([-1, 1])
        self.ball_vy = math.sin(angle) * speed
        self.speed = speed

        self.trail = []
        self.shake = 8

        for _ in range(25):
            self.sparks.append(Spark(self.ball_x, self.ball_y, COL_DEAD))

    def _reset(self, full_reset=True):
        if full_reset:
            self.score = 0
            self.lives = START_LIVES
            self.combo = 0
            self.best_combo = 0
            self.combo_flash = 0
            self.combo_message = ""
            self.combo_message_timer = 0
            self.sparks = []
            self.trail = []
            self.shake = 0
            self.floaters = []
        else:
            if not hasattr(self, "sparks"):
                self.sparks = []
            if not hasattr(self, "trail"):
                self.trail = []
            if not hasattr(self, "shake"):
                self.shake = 0
            if not hasattr(self, "floaters"):
                self.floaters = []

        self.ball_x = float(W // 2)
        self.ball_y = float(H // 4)
        speed = self.ball_speed_0
        angle = random.uniform(math.pi * 0.3, math.pi * 0.7)
        self.ball_vx = math.cos(angle) * speed * random.choice([-1, 1])
        self.ball_vy = math.sin(angle) * speed
        self.speed = speed

        self.paddle_x = W // 2 - self.paddle_w // 2
        self.paddle_x_tgt = self.paddle_x

    # ---------------- GAME STATE ----------------
    def _start_countdown(self):
        self.state = "countdown"
        self.countdown_start = time.time()

    def _toggle_pause(self):
        if self.state == "playing":
            self.state = "paused"
        elif self.state == "paused":
            self.state = "playing"

    def _start_new_game(self):
        self._apply_difficulty(self.selected_difficulty)
        self._reset(full_reset=True)
        self._start_countdown()

    # ---------------- HAND PROCESS ----------------
    def _process_hand(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hand_landmarker.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            self.hand_visible = False
            return

        lm = result.hand_landmarks[0]
        index_tip = lm[8]

        target_x = int(index_tip.x * W)
        target_y = int(index_tip.y * H)

        self.finger_x = int(lerp(self.finger_x, target_x, FINGER_SMOOTHING))
        self.finger_y = int(lerp(self.finger_y, target_y, FINGER_SMOOTHING))
        self.hand_visible = True

    # ---------------- SCORE / COMBO ----------------
    def _combo_bonus(self):
        return self.combo // 5

    def _lose_life(self):
        self.lives -= 1
        self.shake = 15

        if self.combo >= 5:
            self.floaters.append(
                FloatingText(
                    f"Combo broken! x{self.combo}",
                    W // 2 - 120,
                    H // 2 - 20,
                    COL_DEAD,
                    life=1.1,
                    vy=-0.8,
                    scale=0.9,
                )
            )

        self.combo = 0
        self.combo_flash = 0
        self.combo_message = ""
        self.combo_message_timer = 0

        for _ in range(40):
            self.sparks.append(Spark(self.ball_x, self.ball_y, COL_DEAD))

        if self.lives <= 0:
            self.state = "dead"
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()
        else:
            self._reset_ball_only()
            self._start_countdown()

    def _register_hit(self):
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo)
        self.combo_flash = 12

        gain = 1 + self._combo_bonus()
        self.score += gain

        if gain > 1:
            self.floaters.append(
                FloatingText(
                    f"+{gain}",
                    self.ball_x - 10,
                    self.ball_y - 15,
                    COL_COMBO_GLOW,
                    life=0.9,
                    vy=-1.0,
                    scale=0.85,
                )
            )
        else:
            self.floaters.append(
                FloatingText(
                    f"+{gain}",
                    self.ball_x - 8,
                    self.ball_y - 10,
                    COL_SCORE,
                    life=0.8,
                    vy=-0.9,
                    scale=0.75,
                )
            )

        if self.combo in (5, 10, 15, 20):
            self.combo_message = f"COMBO x{self.combo}!"
            self.combo_message_timer = 45
            for _ in range(18):
                self.sparks.append(Spark(self.ball_x, self.ball_y, COL_COMBO_GLOW))

        if self.score > self.high_score:
            self.high_score = self.score
            self.save_high_score()

    # ---------------- UPDATE ----------------
    def _update_countdown(self):
        if self.state != "countdown":
            return

        elapsed = time.time() - self.countdown_start
        if elapsed >= self.countdown_duration + self.go_duration:
            self.state = "playing"

    def _update(self):
        if self.state in ("menu", "dead", "paused", "countdown"):
            self.floaters = [f for f in self.floaters if f.update()]
            self._update_countdown()
            return

        # playing only
        self.paddle_x_tgt = self.finger_x - self.paddle_w // 2
        self.paddle_x_tgt = max(0, min(W - self.paddle_w, self.paddle_x_tgt))
        self.paddle_x += (self.paddle_x_tgt - self.paddle_x) * PADDLE_SMOOTHING

        px = int(self.paddle_x)
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        self.trail.append((int(self.ball_x), int(self.ball_y), 1.0))
        self.trail = [(x, y, a - 0.07) for x, y, a in self.trail if a > 0]

        if self.ball_x - BALL_R <= 0:
            self.ball_x = BALL_R
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + BALL_R >= W:
            self.ball_x = W - BALL_R
            self.ball_vx = -abs(self.ball_vx)

        if self.ball_y - BALL_R <= 0:
            self.ball_y = BALL_R
            self.ball_vy = abs(self.ball_vy)

        ball_bottom = self.ball_y + BALL_R
        paddle_top = PADDLE_Y
        paddle_left = px
        paddle_right = px + self.paddle_w

        if (
            paddle_top - 6 <= ball_bottom <= paddle_top + 12
            and paddle_left - BALL_R <= self.ball_x <= paddle_right + BALL_R
            and self.ball_vy > 0
        ):
            hit_pos = (self.ball_x - (paddle_left + self.paddle_w / 2)) / (self.paddle_w / 2)
            self.ball_vy = -abs(self.ball_vy)
            self.ball_vx += hit_pos * 3.5

            max_speed = self.speed + self.speed_inc * (self.score + 1)
            total = math.hypot(self.ball_vx, self.ball_vy)
            if total > 0:
                self.ball_vx = self.ball_vx / total * max_speed
                self.ball_vy = self.ball_vy / total * max_speed

            self.ball_y = PADDLE_Y - BALL_R - 1
            self._register_hit()
            self.speed = self.ball_speed_0 + self.speed_inc * self.score

            hit_color = COL_COMBO_GLOW if self.combo >= 5 else COL_BALL
            for _ in range(20):
                self.sparks.append(Spark(self.ball_x, self.ball_y, hit_color))
            self.shake = 6

        if self.ball_y - BALL_R > H + 20:
            self._lose_life()

        self.sparks = [s for s in self.sparks if s.update()]
        self.floaters = [f for f in self.floaters if f.update()]

        if self.shake > 0:
            self.shake -= 1
        if self.combo_flash > 0:
            self.combo_flash -= 1
        if self.combo_message_timer > 0:
            self.combo_message_timer -= 1

    # ---------------- DRAW HELPERS ----------------
    def _draw_hearts(self, frame):
        heart_y = 28
        spacing = 42
        start_x = W - 40 - (START_LIVES - 1) * spacing

        for i in range(self.lives):
            x = start_x + i * spacing
            cv2.circle(frame, (x - 8, heart_y), 10, COL_LIVES, -1, cv2.LINE_AA)
            cv2.circle(frame, (x + 8, heart_y), 10, COL_LIVES, -1, cv2.LINE_AA)
            pts = np.array([[x - 18, heart_y + 2], [x + 18, heart_y + 2], [x, heart_y + 26]], np.int32)
            cv2.fillConvexPoly(frame, pts, COL_LIVES)

    def _draw_combo_meter(self, frame):
        combo_txt = f"COMBO x{self.combo}"
        bx, by = 30, 145

        glow = self.combo_flash * 2 if self.combo_flash > 0 else 0
        combo_color = (
            min(255, COL_COMBO[0] + glow),
            min(255, COL_COMBO[1] + glow),
            min(255, COL_COMBO[2] + glow),
        )

        cv2.putText(frame, combo_txt, (bx, by), FONT, 0.9, (20, 20, 30), 5, cv2.LINE_AA)
        cv2.putText(frame, combo_txt, (bx, by), FONT, 0.9, combo_color, 2, cv2.LINE_AA)

        bonus = self._combo_bonus()
        bonus_txt = f"Hit bonus: +{bonus}" if bonus > 0 else "Hit bonus: +0"
        cv2.putText(frame, bonus_txt, (30, 175), FONT, 0.55, (20, 20, 30), 4, cv2.LINE_AA)
        cv2.putText(frame, bonus_txt, (30, 175), FONT, 0.55, (220, 220, 255), 1, cv2.LINE_AA)

        if self.combo_message_timer > 0 and self.combo_message:
            scale = 1.0 + 0.15 * math.sin(time.time() * 10)
            tw = cv2.getTextSize(self.combo_message, FONT, scale, 3)[0][0]
            y = H // 2 - 120
            cv2.putText(frame, self.combo_message, (W // 2 - tw // 2, y), FONT, scale, (30, 20, 40), 8, cv2.LINE_AA)
            cv2.putText(frame, self.combo_message, (W // 2 - tw // 2, y), FONT, scale, COL_COMBO_GLOW, 3, cv2.LINE_AA)

    def _draw_difficulty_info(self, frame):
        txt = f"MODE: {self.difficulty_label}"
        cv2.putText(frame, txt, (W - 260, 110), FONT, 0.65, (20, 25, 35), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (W - 260, 110), FONT, 0.65, self.difficulty_color, 2, cv2.LINE_AA)

    def _draw_countdown_overlay(self, frame):
        if self.state != "countdown":
            return

        elapsed = time.time() - self.countdown_start
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (10, 8, 25), -1)
        cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)

        if elapsed < 1.0:
            txt = "3"
            color = (255, 220, 120)
        elif elapsed < 2.0:
            txt = "2"
            color = (255, 200, 120)
        elif elapsed < 3.0:
            txt = "1"
            color = (255, 170, 120)
        else:
            txt = "GO!"
            color = (120, 255, 180)

        scale = 3.2 if txt != "GO!" else 2.4
        thickness = 6
        tw = cv2.getTextSize(txt, FONT, scale, thickness)[0][0]
        cv2.putText(frame, txt, (W // 2 - tw // 2, H // 2 + 30), FONT, scale, (30, 30, 40), thickness + 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (W // 2 - tw // 2, H // 2 + 30), FONT, scale, color, thickness, cv2.LINE_AA)

    def _draw_pause_overlay(self, frame):
        if self.state != "paused":
            return

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (8, 6, 18), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        title = "PAUSED"
        tw = cv2.getTextSize(title, FONT, 2.2, 4)[0][0]
        cv2.putText(frame, title, (W // 2 - tw // 2, H // 2 - 20), FONT, 2.2, (30, 30, 40), 8, cv2.LINE_AA)
        cv2.putText(frame, title, (W // 2 - tw // 2, H // 2 - 20), FONT, 2.2, COL_PAUSE, 4, cv2.LINE_AA)

        hint = "Press P to resume"
        hw = cv2.getTextSize(hint, FONT, 0.75, 2)[0][0]
        cv2.putText(frame, hint, (W // 2 - hw // 2, H // 2 + 45), FONT, 0.75, (220, 230, 255), 2, cv2.LINE_AA)

    # ---------------- DRAW MAIN ----------------
    def _draw(self, frame):
        t = time.time()
        ox = random.randint(-self.shake, self.shake) if self.shake else 0
        oy = random.randint(-self.shake, self.shake) if self.shake else 0

        frame[:] = np.roll(self.cam_frame, (ox, oy), axis=(1, 0))

        for tx, ty, alpha in self.trail:
            r = max(2, int(BALL_R * 0.5 * alpha))
            c = tuple(int(ch * alpha * 0.7) for ch in COL_TRAIL)
            cv2.circle(frame, (tx + ox, ty + oy), r, c, -1, cv2.LINE_AA)

        for s in self.sparks:
            s.draw(frame)
        for f in self.floaters:
            f.draw(frame)

        if self.state in ("playing", "paused", "countdown", "dead"):
            px = int(self.paddle_x) + ox
            paddle_col = COL_COMBO_GLOW if self.combo >= 10 else COL_PADDLE
            draw_glow_rect(frame, px, PADDLE_Y + oy, self.paddle_w, PADDLE_H, paddle_col)

            bx, by = int(self.ball_x) + ox, int(self.ball_y) + oy
            if self.state != "dead":
                ball_col = COL_COMBO_GLOW if self.combo >= 10 else COL_BALL
                draw_glow_circle(frame, bx, by, BALL_R, ball_col)
                cv2.circle(frame, (bx, by), BALL_R, ball_col, -1, cv2.LINE_AA)
                cv2.circle(frame, (bx - 5, by - 5), BALL_R // 3, (220, 255, 230), -1, cv2.LINE_AA)

            danger_alpha = 0.18 + 0.12 * abs(math.sin(t * 4))
            ov = frame.copy()
            cv2.line(ov, (0, PADDLE_Y + PADDLE_H + 30), (W, PADDLE_Y + PADDLE_H + 30), (50, 60, 255), 2)
            cv2.addWeighted(ov, danger_alpha, frame, 1 - danger_alpha, 0, frame)

            score_txt = str(self.score)
            cv2.putText(frame, score_txt, (28, 60), FONT, 1.8, (30, 40, 60), 5, cv2.LINE_AA)
            cv2.putText(frame, score_txt, (28, 60), FONT, 1.8, COL_SCORE, 3, cv2.LINE_AA)
            cv2.putText(frame, "SCORE", (30, 82), FONT, 0.40, (140, 160, 200), 1, cv2.LINE_AA)

            hs = f"BEST: {self.high_score}"
            cv2.putText(frame, hs, (30, 112), FONT, 0.72, (20, 25, 35), 4, cv2.LINE_AA)
            cv2.putText(frame, hs, (30, 112), FONT, 0.72, COL_HIGHSCORE, 2, cv2.LINE_AA)

            self._draw_combo_meter(frame)

            cv2.putText(frame, "LIVES", (W - 180, 70), FONT, 0.60, (20, 25, 35), 4, cv2.LINE_AA)
            cv2.putText(frame, "LIVES", (W - 180, 70), FONT, 0.60, COL_LIVES, 2, cv2.LINE_AA)
            self._draw_hearts(frame)

            self._draw_difficulty_info(frame)

            if self.hand_visible and self.state in ("playing", "paused", "countdown"):
                fx, fy = self.finger_x, self.finger_y
                cv2.circle(frame, (fx, fy), 10, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (fx, fy), 4, (255, 255, 255), -1, cv2.LINE_AA)

        if self.state == "menu":
            self._draw_menu(frame, t)
        elif self.state == "dead":
            self._draw_dead(frame, t)

        self._draw_countdown_overlay(frame)
        self._draw_pause_overlay(frame)

    # ---------------- MENU / OVERLAYS ----------------
    def _draw_menu(self, frame, t):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (8, 4, 18), -1)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

        pulse = int(abs(math.sin(t * 2)) * 25)
        title = "BALL CATCH"
        tw = cv2.getTextSize(title, FONT, 2.8, 5)[0][0]

        for thickness, alpha_val in [(18, 0.05), (10, 0.12), (5, 0.3)]:
            ov2 = frame.copy()
            cv2.putText(ov2, title, (W // 2 - tw // 2, 150), FONT, 2.8, COL_PADDLE, thickness, cv2.LINE_AA)
            cv2.addWeighted(ov2, alpha_val, frame, 1 - alpha_val, 0, frame)

        cv2.putText(frame, title, (W // 2 - tw // 2, 150), FONT, 2.8, COL_PADDLE, 4, cv2.LINE_AA)

        sub = "Choose difficulty before starting"
        sw = cv2.getTextSize(sub, FONT, 0.80, 2)[0][0]
        cv2.putText(frame, sub, (W // 2 - sw // 2, 220), FONT, 0.80, (220, 235, 255), 2, cv2.LINE_AA)

        # difficulty menu box
        box_w, box_h = 520, 240
        box_x = W // 2 - box_w // 2
        box_y = 270
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 18, 35), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 140, 220), 2)

        entries = [
            ("1  -  EASY", "easy"),
            ("2  -  NORMAL", "normal"),
            ("3  -  HARD", "hard"),
        ]

        start_y = box_y + 65
        gap = 58
        for i, (label, key_name) in enumerate(entries):
            y = start_y + i * gap
            selected = self.selected_difficulty == key_name
            cfg = DIFFICULTIES[key_name]
            color = cfg["color"] if selected else (180, 190, 220)
            scale = 0.95 if selected else 0.82
            thickness = 3 if selected else 2

            if selected:
                sel_overlay = frame.copy()
                cv2.rectangle(sel_overlay, (box_x + 25, y - 32), (box_x + box_w - 25, y + 16), (40, 40, 80), -1)
                cv2.addWeighted(sel_overlay, 0.50, frame, 0.50, 0, frame)

                pointer = ">"
                cv2.putText(frame, pointer, (box_x + 38, y), FONT, 1.0, color, 3, cv2.LINE_AA)

            cv2.putText(frame, label, (box_x + 80, y), FONT, scale, (20, 20, 30), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, label, (box_x + 80, y), FONT, scale, color, thickness, cv2.LINE_AA)

        current = f"Selected: {DIFFICULTIES[self.selected_difficulty]['label']}"
        cw = cv2.getTextSize(current, FONT, 0.72, 2)[0][0]
        cv2.putText(frame, current, (W // 2 - cw // 2, 555), FONT, 0.72, DIFFICULTIES[self.selected_difficulty]["color"], 2, cv2.LINE_AA)

        hint = "Press 1 / 2 / 3 to choose difficulty"
        hw = cv2.getTextSize(hint, FONT, 0.62, 1)[0][0]
        cv2.putText(frame, hint, (W // 2 - hw // 2, 610), FONT, 0.62, (200, 210, 240), 1, cv2.LINE_AA)

        hint2 = "Press SPACE to start   •   ESC to quit"
        hw2 = cv2.getTextSize(hint2, FONT, 0.62, 1)[0][0]
        cv2.putText(frame, hint2, (W // 2 - hw2 // 2, 650), FONT, 0.62, (int(150 + pulse), int(170 + pulse), int(230 + pulse)), 1, cv2.LINE_AA)

        best_txt = f"Best Score: {self.high_score}"
        bw = cv2.getTextSize(best_txt, FONT, 0.65, 2)[0][0]
        cv2.putText(frame, best_txt, (W // 2 - bw // 2, 695), FONT, 0.65, COL_HIGHSCORE, 2, cv2.LINE_AA)

    def _draw_dead(self, frame, t):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (5, 3, 15), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        pulse = int(abs(math.sin(t * 3)) * 30)
        go_txt = "GAME OVER"
        tw = cv2.getTextSize(go_txt, FONT, 2.6, 5)[0][0]
        go_col = (int(80 + pulse), int(70 + pulse), 255)
        cv2.putText(frame, go_txt, (W // 2 - tw // 2, H // 2 - 120), FONT, 2.6, go_col, 5, cv2.LINE_AA)

        sc_txt = f"Final Score: {self.score}"
        sw = cv2.getTextSize(sc_txt, FONT, 0.95, 2)[0][0]
        cv2.putText(frame, sc_txt, (W // 2 - sw // 2, H // 2 - 25), FONT, 0.95, (220, 235, 255), 2, cv2.LINE_AA)

        hs_txt = f"Best Score: {self.high_score}"
        hw = cv2.getTextSize(hs_txt, FONT, 0.78, 2)[0][0]
        cv2.putText(frame, hs_txt, (W // 2 - hw // 2, H // 2 + 20), FONT, 0.78, COL_HIGHSCORE, 2, cv2.LINE_AA)

        combo_txt = f"Best Combo: x{self.best_combo}"
        cw = cv2.getTextSize(combo_txt, FONT, 0.72, 2)[0][0]
        cv2.putText(frame, combo_txt, (W // 2 - cw // 2, H // 2 + 60), FONT, 0.72, COL_COMBO_GLOW, 2, cv2.LINE_AA)

        mode_txt = f"Mode: {self.difficulty_label}"
        mw = cv2.getTextSize(mode_txt, FONT, 0.66, 2)[0][0]
        cv2.putText(frame, mode_txt, (W // 2 - mw // 2, H // 2 + 100), FONT, 0.66, (220, 220, 255), 2, cv2.LINE_AA)

        restart = "SPACE = play again   |   M = main menu   |   ESC = quit"
        rw = cv2.getTextSize(restart, FONT, 0.52, 1)[0][0]
        cv2.putText(frame, restart, (W // 2 - rw // 2, H // 2 + 150), FONT, 0.52, (140, 160, 200), 1, cv2.LINE_AA)

    # ---------------- MAIN LOOP ----------------
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        cv2.namedWindow("Ball Catch", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ball Catch", W, H)

        frame_out = np.zeros((H, W, 3), dtype=np.uint8)

        while True:
            ret, cam = cap.read()
            if not ret:
                cam = np.zeros((H, W, 3), dtype=np.uint8)

            cam = cv2.resize(cam, (W, H))
            cam = cv2.flip(cam, 1)
            self.cam_frame = cam.copy()

            self._process_hand(cam)
            self._update()
            self._draw(frame_out)
            cv2.imshow("Ball Catch", frame_out)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # difficulty selection (menu or dead screen)
            if key == ord('1'):
                self.selected_difficulty = "easy"
            elif key == ord('2'):
                self.selected_difficulty = "normal"
            elif key == ord('3'):
                self.selected_difficulty = "hard"

            # if playing-related state, allow live difficulty switch only when not actively playing
            if self.state in ("menu", "dead"):
                self._apply_difficulty(self.selected_difficulty)

            # pause toggle
            if key == ord('p') or key == ord('P'):
                if self.state in ("playing", "paused"):
                    self._toggle_pause()

            # main menu from game over
            if key == ord('m') or key == ord('M'):
                if self.state == "dead":
                    self.state = "menu"

            # start game from menu or restart from dead
            if key == ord(' '):
                if self.state in ("menu", "dead"):
                    self._start_new_game()

        cap.release()
        cv2.destroyAllWindows()

    def close(self):
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()


if __name__ == "__main__":
    game = BallCatchGame()
    try:
        game.run()
    finally:
        game.close()