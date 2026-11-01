import os
import cv2
import numpy as np
import pyautogui
import time
import keyboard
import threading

import onnxruntime as ort

MODEL_PATH = "bestV1.onnx"

# ===== Загрузка ONNX-модели =====
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Индексы -> имена классов (порядок как в data.yaml)
CLASS_NAMES = ["game", "upgrade", "reclama"]

# В модели класс №2 назван "advertising" — приводим его к нашему имени "reclama"
CLASS_ALIASES = {
    "advertising": "reclama",
    "reklama": "reclama",
    "ad": "reclama",
}


def normalize_name(name):
    return CLASS_ALIASES.get(name, name)


TARGET_CLASSES = ("game", "upgrade", "reclama")

IMGSZ = 640
NMS_THRESHOLD = 0.45

pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True

running = False
game_pos = None          # текущая позиция game (обновляется из основного цикла)
mouse_lock = threading.Lock()


def toggle():
    global running
    running = not running
    print(">>>", "ВКЛ" if running else "ВЫКЛ")


keyboard.add_hotkey("space", toggle)

# ===== Настройка уверенности (порог детекции) =====
CONF_FILE = "conf.txt"


def load_conf(default=0.5):
    """Читает CONF из conf.txt. Если файла нет — из config.yaml, иначе default."""
    try:
        with open(CONF_FILE, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        pass
    try:
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get("confidence_threshold", default))
    except Exception:
        return default


def save_conf(value):
    try:
        with open(CONF_FILE, "w", encoding="utf-8") as f:
            f.write(f"{value:.2f}")
    except Exception as e:
        print("Не удалось сохранить conf.txt:", e)


CONF = load_conf()
CONF_STEP = 0.05
CONF_MIN = 0.05
CONF_MAX = 0.95


def conf_up():
    global CONF
    CONF = min(CONF_MAX, round(CONF + CONF_STEP, 2))
    save_conf(CONF)
    print(f">>> уверенность (CONF) = {CONF:.2f}")


def conf_down():
    global CONF
    CONF = max(CONF_MIN, round(CONF - CONF_STEP, 2))
    save_conf(CONF)
    print(f">>> уверенность (CONF) = {CONF:.2f}")


# '+'/'-' (и '='/'_' как запасные) — поднять/опустить порог уверенности на ходу
keyboard.add_hotkey("+", conf_up)
keyboard.add_hotkey("=", conf_up)
keyboard.add_hotkey("-", conf_down)
keyboard.add_hotkey("_", conf_down)

SLEEP = 0.1

# Шаг вертикального прохода по боксу upgrade (в пикселях)
UPGRADE_STEP = 5

# Смещение курсора по upgrade вправо (в пикселях)
UPGRADE_X_OFFSET = 20

# Как часто отдельный поток кликает по game (сек) — меньше = чаще
GAME_CLICK_INTERVAL = 0.03


def letterbox(frame, new_shape=640, color=114):
    """Ресайз с сохранением пропорций + паддинг до квадрата 640x640 (top-left)."""
    h, w = frame.shape[:2]
    scale = new_shape / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def infer(canvas):
    """Гоняет ONNX-модель и возвращает список детекций (cls_id, conf, x1, y1, x2, y2)."""
    # BGR -> RGB, HWC -> CHW, normalize 0..1, add batch dim
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # (3, 640, 640)
    img = np.expand_dims(img, axis=0)           # (1, 3, 640, 640)

    out = session.run([output_name], {input_name: img})[0]  # (1, 7, 8400)

    # (1, 7, 8400) -> (8400, 7)
    pred = out[0].transpose(1, 0)

    boxes = []
    scores = []
    class_ids = []

    # Первые 4 значения — cx, cy, w, h; далее — скоры классов
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]

    confidences = cls_scores.max(axis=1)
    class_ids_all = cls_scores.argmax(axis=1)

    mask = confidences >= CONF
    for i in np.where(mask)[0]:
        cx, cy, bw, bh = boxes_xywh[i]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        boxes.append([float(x1), float(y1), float(bw), float(bh)])
        scores.append(float(confidences[i]))
        class_ids.append(int(class_ids_all[i]))

    if not boxes:
        return []

    # NMS отдельно по каждому классу (чтобы upgrade и game не глушили друг друга)
    detections = []
    for cid in set(class_ids):
        idxs = [j for j, c in enumerate(class_ids) if c == cid]
        b = [boxes[j] for j in idxs]
        s = [scores[j] for j in idxs]
        keep = cv2.dnn.NMSBoxes(b, s, CONF, NMS_THRESHOLD)
        if len(keep) == 0:
            continue
        keep = np.array(keep).flatten()
        for k in keep:
            j = idxs[int(k)]
            detections.append((class_ids[j], scores[j],
                               boxes[j][0], boxes[j][1],
                               boxes[j][0] + boxes[j][2],
                               boxes[j][1] + boxes[j][3]))
    return detections


def game_clicker_loop():
    """Кликает по game ПАРАЛЛЕЛЬНО с проходом по upgrade."""
    while True:
        pos = game_pos
        if running and pos is not None:
            with mouse_lock:
                pyautogui.click(pos[0], pos[1])
            time.sleep(GAME_CLICK_INTERVAL)
        else:
            time.sleep(0.01)


# Отдельный поток для частых кликов по game
threading.Thread(target=game_clicker_loop, daemon=True).start()

print("ПРОБЕЛ — старт/стоп кликера. Esc — выход.")
print(f"Настройка уверенности: '+' повысить, '-' понизить (текущая CONF = {CONF:.2f}, сохраняется в conf.txt)")
print("Целевые классы: game, upgrade, reclama")

while True:
    if not running:
        time.sleep(0.05)
        continue

    # Скриншот экрана
    screen = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]

    # >>> РЕСАЙЗ ПОД IMGSZ + ПАДДИНГ (letterbox)
    canvas, scale = letterbox(frame, IMGSZ)

    # Инференс на 640x640
    detections = infer(canvas)

    # Собираем ВСЕ целевые детекции (не только первую!)
    targets = []
    game_pos = None  # сброс: если game не найден, поток перестанет кликать
    for cls_id, conf, x1, y1, x2, y2 in detections:
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue
        name = normalize_name(CLASS_NAMES[cls_id])
        if name not in TARGET_CLASSES:
            continue

        # >>> ОБРАТНЫЙ МАСШТАБ БОКСОВ В ЭКРАННЫЕ КООРДИНАТЫ
        x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)

        # Страховка: не выходим за пределы экрана
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        targets.append((name, conf, x1, y1, x2, y2))

    # Приоритет: сначала game, затем upgrade, затем reclama (по убыванию conf внутри класса)
    priority = {"game": 0, "upgrade": 1, "reclama": 2}
    targets.sort(key=lambda t: (priority[t[0]], -t[1]))

    for name, conf, x1, y1, x2, y2 in targets:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if name == "game":
            # Позицию game отдаём отдельному потоку — он кликает часто и параллельно
            game_pos = (cx, cy)
        elif name == "upgrade":
            # Кликаем по всему боксу сверху вниз, чуть правее центра
            cx = min(cx + UPGRADE_X_OFFSET, w - 1)
            print(f"[{time.strftime('%H:%M:%S')}] upgrade ({conf:.2f}) бокс ({x1},{y1})-({x2},{y2})")
            y = y1
            while y <= y2:
                with mouse_lock:
                    pyautogui.moveTo(cx, y, duration=0.02)
                    time.sleep(0.01)
                    pyautogui.click()
                time.sleep(0.02)
                y += UPGRADE_STEP
        elif name == "reclama":
            print(f"[{time.strftime('%H:%M:%S')}] reclama ({conf:.2f}) в ({cx}, {cy})")
            with mouse_lock:
                pyautogui.moveTo(cx, cy, duration=0.05)
                time.sleep(0.02)
                pyautogui.click()
            time.sleep(0.05)

    time.sleep(SLEEP)
