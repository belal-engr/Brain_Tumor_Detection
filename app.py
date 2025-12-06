import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from pathlib import Path
import threading
import time
import random

# -----------------------------
# PATH CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / r"D:\brain tumor\YOLOv8\tumor_detection\weights\best.pt"

# Load YOLO model
model = YOLO(str(MODEL_PATH))

stop_webcam = False
animation_running = False

# -----------------------------
# MATRIX / MEDICAL THEME COLORS
# -----------------------------
BG = "#0b0202"
SIDEBAR_BG = "#4D1111"
PANEL_BG = "#420606"
BTN_BG = "#5C0707"
BTN_HOVER = "#861111"
TEXT = "#a31515"
MUTED = "#c40d0d"
ACCENT = "#c73030"
ACCENT_SOFT = "#3a0505"

panel = None
status_label = None
detail_label = None
log_text = None
loader_canvas = None

angle = 0


# =========================================================
#  SYSTEM LOG CONSOLE
# =========================================================
def log(msg: str):
    if log_text is None:
        return
    log_text.config(state="normal")
    prefix = random.choice(["[SCAN]", "[NODE]", "[CORE]", "[SYS]", "[MODEL]"])
    log_text.insert("end", f"{prefix} {msg}\n")
    log_text.see("end")
    log_text.config(state="disabled")


# =========================================================
#  LOADER ANIMATION
# =========================================================
def start_loader():
    global animation_running, loader_canvas
    if animation_running:
        return

    animation_running = True

    loader_canvas = tk.Canvas(
        panel, width=200, height=200,
        bg=PANEL_BG, highlightthickness=0
    )
    loader_canvas.place(relx=0.5, rely=0.5, anchor="center")

    animate_loader()


def stop_loader():
    global animation_running, loader_canvas
    animation_running = False
    if loader_canvas:
        loader_canvas.destroy()
        loader_canvas = None


def animate_loader():
    global angle, animation_running
    if not animation_running or loader_canvas is None:
        return

    loader_canvas.delete("all")
    cx, cy = 100, 100

    loader_canvas.create_arc(
        cx - 60, cy - 60, cx + 60, cy + 60,
        start=angle, extent=120,
        style="arc", width=6, outline=ACCENT
    )

    loader_canvas.create_text(
        cx, cy + 50, text="SCANNING...",
        fill=ACCENT, font=("Consolas", 10, "bold")
    )

    angle = (angle + 10) % 360
    loader_canvas.after(40, animate_loader)


# -----------------------------
# BUTTON HOVER EFFECT
# -----------------------------
def on_enter(e):
    e.widget["background"] = BTN_HOVER


def on_leave(e):
    e.widget["background"] = BTN_BG


# -----------------------------
# SHOW IMAGE
# -----------------------------
def show_image(img_path: Path):
    try:
        img = Image.open(img_path).resize((800, 500))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image:\n{e}")
        log(f"ERROR loading image -> {e}")


# -----------------------------
# IMAGE DETECTION (FIXED)
# -----------------------------
def detect_image():
    file_path = filedialog.askopenfilename(
        title="Select MRI Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    log(f"Image selected: {file_path}")
    status_label.config(text="STATUS: MRI IMAGE ANALYSIS")
    detail_label.config(text="DETAIL: Tumor detection initialized...")
    start_loader()
    root.update_idletasks()

    try:
        results = model.predict(file_path, save=True)
        r0 = results[0]
        save_dir = Path(r0.save_dir)

        stem = Path(file_path).stem
        candidates = list(save_dir.glob(f"{stem}.*"))

        output_img = candidates[0] if candidates else r0.path
        show_image(output_img)
        log("Brain tumor analysis complete.")

        # -----------------------------
        # FIXED DETECTION LOGIC
        # -----------------------------
        detected = False
        confidence = 0

        for r in results:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    detected = True
                    confidence = round(conf * 100, 2)
                    break

        if not detected:
            status_label.config(text="STATUS: NO TUMOR DETECTED")
            detail_label.config(text="DETAIL: MRI scan shows no tumor.")
            log("NO tumor detected.")
        else:
            status_label.config(text="STATUS: TUMOR DETECTED")
            detail_label.config(text=f"DETAIL: CONFIDENCE={confidence}%")
            log(f"Tumor detected (Confidence: {confidence}%)")

            messagebox.showwarning(
                "Detection Result",
                f"Brain Tumor Detected!\nConfidence: {confidence}%"
            )

    except Exception as e:
        messagebox.showerror("Error", f"Detection failed:\n{e}")
        log(f"ERROR image detection -> {e}")

    finally:
        stop_loader()


# -----------------------------
# VIDEO DETECTION
# -----------------------------
def detect_video():
    file_path = filedialog.askopenfilename(
        title="Select MRI Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
    )
    if not file_path:
        return

    log(f"Video selected: {file_path}")
    status_label.config(text="STATUS: MRI VIDEO ANALYSIS")
    detail_label.config(text="DETAIL: Processing frames...")
    start_loader()
    root.update_idletasks()

    try:
        model.predict(source=file_path, save=True,
                      project="runs/detect", name="video_mri", exist_ok=True)

        out = Path("runs/detect/video_mri") / f"{Path(file_path).stem}.mp4"
        messagebox.showinfo("Done!", f"Video saved at:\n{out}")

        status_label.config(text="STATUS: MRI VIDEO COMPLETE")
        detail_label.config(text="DETAIL: Saved to runs/detect/video_mri")
        log(f"Video processing complete: {out}")

    except Exception as e:
        messagebox.showerror("Error", f"Video detection failed:\n{e}")
        log(f"ERROR video detection -> {e}")

    finally:
        stop_loader()


# -----------------------------
# LIVE DEMO (Webcam)
# -----------------------------
def run_webcam():
    global stop_webcam
    stop_webcam = False

    status_label.config(text="STATUS: LIVE MRI DEMO")
    detail_label.config(text="DETAIL: Webcam active...")
    log("LIVE MODE: Webcam initializing...")

    start_loader()
    root.update_idletasks()
    time.sleep(1)
    stop_loader()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not found.")
        return

    log("Webcam online.")

    try:
        while not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            frame = results.plot()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((800, 500))
            img_tk = ImageTk.PhotoImage(img)
            panel.config(image=img_tk)
            panel.image = img_tk

    finally:
        cap.release()
        stop_webcam = True
        status_label.config(text="STATUS: LIVE FEED TERMINATED")
        detail_label.config(text="DETAIL: Stream closed.")


def start_webcam():
    thread = threading.Thread(target=run_webcam)
    thread.daemon = True
    thread.start()


def stop_webcam_func():
    global stop_webcam
    stop_webcam = True
    log("Command received: Terminate feed")


# -----------------------------
# UI BUILDER
# -----------------------------
root = tk.Tk()
root.title("BRAIN TUMOR DETECTION // YOLOv8 MEDICAL SCAN")

root.state("zoomed")
root.configure(bg=BG)

header = tk.Frame(root, bg=BG)
header.pack(fill="x", pady=(10, 0), padx=15)

tk.Label(
    header, text="BRAIN TUMOR DETECTION CONSOLE",
    fg=ACCENT, bg=BG, font=("Consolas", 20, "bold")
).pack(side="left")

tk.Label(
    header,
    text="MODE: MEDICAL ANALYSIS  //  ENGINE: YOLOv8  //  STATUS: READY",
    fg=MUTED, bg=BG, font=("Consolas", 9)
).pack(side="left", padx=(15, 0))

tk.Frame(root, bg=ACCENT, height=2).pack(fill="x", padx=15, pady=(8, 10))

main = tk.Frame(root, bg=BG)
main.pack(fill="both", expand=True, padx=15, pady=10)

# Sidebar
sidebar = tk.Frame(main, bg=SIDEBAR_BG, width=280)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

tk.Label(
    sidebar, text="> CONTROL PANEL",
    fg=ACCENT, bg=SIDEBAR_BG, font=("Consolas", 14, "bold")
).pack(pady=(20, 5), anchor="w", padx=20)

btn_box = tk.Frame(sidebar, bg=SIDEBAR_BG)
btn_box.pack(padx=20, pady=10, fill="x")

def make_btn(text, command):
    btn = tk.Button(
        btn_box, text=text, command=command,
        fg=TEXT, bg=BTN_BG, activebackground=BTN_HOVER,
        bd=0, relief="flat", font=("Consolas", 11), pady=6
    )
    btn.pack(fill="x", pady=6)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

make_btn("▣  MRI IMAGE SCAN", detect_image)
make_btn("▣  MRI VIDEO SCAN", detect_video)
make_btn("▣  LIVE DEMO (Webcam)", start_webcam)
make_btn("✘  TERMINATE FEED", stop_webcam_func)

# LOG PANEL
tk.Label(
    sidebar, text="\n> SYSTEM LOG",
    fg=ACCENT, bg=SIDEBAR_BG, font=("Consolas", 12, "bold")
).pack(anchor="w", padx=20)

log_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
log_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))

log_text = tk.Text(
    log_frame, bg="#000000", fg=ACCENT,
    insertbackground=ACCENT, font=("Consolas", 9),
    relief="flat", wrap="none"
)
log_text.pack(fill="both", expand=True)
log_text.config(state="disabled")

log("SYSTEM ONLINE: Brain Tumor Detection Ready.")
log("Model Loaded Successfully.")

# PREVIEW PANEL
content = tk.Frame(main, bg=BG)
content.pack(side="left", fill="both", expand=True, padx=(20, 0))

preview_outer = tk.Frame(content, bg=ACCENT_SOFT)
preview_outer.pack(pady=(5, 10))

preview_inner = tk.Frame(preview_outer, bg=PANEL_BG)
preview_inner.pack(padx=2, pady=2)

panel = tk.Label(preview_inner, bg=PANEL_BG, width=800, height=500)
panel.pack()

status_frame = tk.Frame(content, bg=BG)
status_frame.pack(fill="x")

status_label = tk.Label(
    status_frame, text="STATUS: IDLE",
    fg=ACCENT, bg=BG, font=("Consolas", 11, "bold")
)
status_label.pack(fill="x")

detail_label = tk.Label(
    status_frame, text="DETAIL: Awaiting medical scan request...",
    fg=MUTED, bg=BG, font=("Consolas", 9)
)
detail_label.pack(fill="x")

root.mainloop()
