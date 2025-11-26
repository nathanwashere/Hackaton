import random
import time
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox

# =====================================
# "מילון" לוגים של Bluetooth
# =====================================

# תבניות ללוגים נורמליים של BT (בערך 80%)
BT_NORMAL_TEMPLATES = [
    "BT NORMAL | {ts} | Connected to device '{device}' ({mac}), RSSI {rssi} dBm",
    "BT NORMAL | {ts} | Audio streaming active from '{device}' ({mac})",
    "BT NORMAL | {ts} | Device '{device}' ({mac}) sent control command: {cmd}",
    "BT NORMAL | {ts} | Pairing successful with '{device}' ({mac})",
    "BT NORMAL | {ts} | Hands-free call started with '{device}' ({mac})",
    "BT NORMAL | {ts} | Device '{device}' ({mac}) idle, link OK (RSSI {rssi} dBm)",
]

# תבניות ללוגים חריגים של BT (בערך 20%)
BT_ANOMALY_TEMPLATES = [
    "BT ANOMALY | {ts} | Multiple pairing attempts from unknown device ({mac}) – POSSIBLE BRUTE FORCE",
    "BT ANOMALY | {ts} | RSSI spike from ({mac}) with unusual traffic pattern – POSSIBLE NEARBY ATTACKER",
    "BT ANOMALY | {ts} | Device '{device}' ({mac}) trying to access restricted API – BLOCKED",
    "BT ANOMALY | {ts} | Suspicious data volume from '{device}' ({mac}) – POTENTIAL EXPLOIT",
]

BT_SAMPLE_DEVICES = [
    "Galaxy S23",
    "iPhone 15",
    "Pixel 9",
    "Unknown_Device",
]

BT_SAMPLE_COMMANDS = [
    "PLAY",
    "PAUSE",
    "NEXT_TRACK",
    "PREV_TRACK",
    "VOLUME_UP",
    "VOLUME_DOWN",
]

# =====================================
# סימולציה פשוטה של לוגים ל-CANBUS
# (רק בשביל עיניים – לא קשור ל-ML.py)
# =====================================

CAN_TEMPLATES = [
    "CAN | {ts} | ID=0x{can_id:03X} | Speed={speed:5.1f} km/h | Brake={brake:4.1f}% | ABS={abs_state}",
    "CAN | {ts} | ID=0x{can_id:03X} | WheelSpeedFL={wfl:5.1f} km/h | WheelSpeedFR={wfr:5.1f} km/h",
    "CAN | {ts} | ID=0x{can_id:03X} | ECU=ABS | Status={abs_state} | Pressure={brake:4.1f} bar",
]


# =====================================
# פונקציות יצירת לוג אחד (BT + CAN)
# =====================================

def current_ts():
    """טיים־סטמפ קצר לקריאה."""
    return time.strftime("%H:%M:%S")


def generate_bt_log():
    """
    מחזיר tuple:
      (log_line: str, is_anomaly: bool)
    עם התפלגות 80% נורמלי / 20% חריג.
    """
    is_anomaly = random.random() < 0.20  # 20% חריג
    device = random.choice(BT_SAMPLE_DEVICES)
    mac = "AA:BB:{:02X}:{:02X}:{:02X}:{:02X}".format(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )
    rssi = random.randint(-80, -40)
    cmd = random.choice(BT_SAMPLE_COMMANDS)
    ts = current_ts()

    if not is_anomaly:
        template = random.choice(BT_NORMAL_TEMPLATES)
    else:
        template = random.choice(BT_ANOMALY_TEMPLATES)

    line = template.format(
        ts=ts,
        device=device,
        mac=mac,
        rssi=rssi,
        cmd=cmd,
    )
    return line, is_anomaly


def generate_can_log():
    """
    מחזיר שורת לוג מדומה של CANBUS.
    נטו בשביל העין – לא קשור לקוד של ה-ML.
    """
    ts = current_ts()
    can_id = random.randint(0x100, 0x1FF)
    speed = random.uniform(0, 130)
    brake = random.uniform(0, 100)
    abs_state = random.choice(["OFF", "ON", "ACTIVE"])
    wfl = max(0.0, speed + random.uniform(-5, 3))
    wfr = max(0.0, speed + random.uniform(-5, 3))

    template = random.choice(CAN_TEMPLATES)
    line = template.format(
        ts=ts,
        can_id=can_id,
        speed=speed,
        brake=brake,
        abs_state=abs_state,
        wfl=wfl,
        wfr=wfr,
    )
    return line


# =====================================
# GUI – שני חלונות לוג + התראה על חריגת BT
# =====================================

class LogGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CAN + Bluetooth Log Monitor (Demo)")
        self.root.geometry("1100x600")

        self.bt_stopped_after_anomaly = False
        self.stop_all_logs = False

        # Frame ראשי
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # חלון לוגים של CAN
        can_frame = tk.Frame(main_frame)
        can_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(can_frame, text="CANBUS Logs", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.can_text = ScrolledText(can_frame, height=30, width=60, state=tk.NORMAL)
        self.can_text.pack(fill=tk.BOTH, expand=True)

        # חלון לוגים של BT
        bt_frame = tk.Frame(main_frame)
        bt_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        tk.Label(bt_frame, text="Bluetooth Logs", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.bt_text = ScrolledText(bt_frame, height=30, width=60, state=tk.NORMAL)
        self.bt_text.pack(fill=tk.BOTH, expand=True)

        # כפתורים למטה
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.clear_btn = tk.Button(btn_frame, text="Clear Logs", command=self.clear_logs)
        self.clear_btn.pack(side=tk.LEFT)

        self.quit_btn = tk.Button(btn_frame, text="Quit", command=root.destroy)
        self.quit_btn.pack(side=tk.RIGHT)

        # התחלת לולאת לוגים
        self.schedule_can_log()
        self.schedule_bt_log()

    # ---------- CAN ----------

    def schedule_can_log(self):
        if self.stop_all_logs:
            return
        """הוספת לוג CAN כל ~200ms."""
        line = generate_can_log()
        self.append_can_log(line)
        # לתנועה יפה בעין
        self.root.after(200, self.schedule_can_log)

    def append_can_log(self, line: str):
        self.can_text.insert(tk.END, line + "\n")
        self.can_text.see(tk.END)

    # ---------- BT ----------

    def schedule_bt_log(self):
        """
        הוספת לוג BT:
        - 80% נורמלי
        - 20% חריג
        אם התקבלה חריגה → נעצור את הזרימה ונציג התראה.
        """
        if self.bt_stopped_after_anomaly:
            return  # לא מוסיף יותר BT אחרי חריגה

        line, is_anomaly = generate_bt_log()
        self.append_bt_log(line, is_anomaly)

        if is_anomaly:
            self.bt_stopped_after_anomaly = True
            self.stop_all_logs = True
            
            messagebox.showwarning(
                "Bluetooth Anomaly Detected",
                "Detected suspicious Bluetooth activity.\n"
                "BT logging stopped for investigation."
            )
            # אפשר גם להוסיף כאן לוג ב-CAN או שינוי מצב וכו'
        else:
            # ממשיכים להזרים BT בשגרה (כל 300ms)
            self.root.after(300, self.schedule_bt_log)

    def append_bt_log(self, line: str, is_anomaly: bool):
        if is_anomaly:
            # נצבע חריגות באדום
            self.bt_text.insert(tk.END, line + "\n", "anomaly")
            self.bt_text.tag_config("anomaly", foreground="red")
        else:
            self.bt_text.insert(tk.END, line + "\n")
        self.bt_text.see(tk.END)

    # ---------- כללי ----------

    def clear_logs(self):
        self.can_text.delete("1.0", tk.END)
        self.bt_text.delete("1.0", tk.END)
        self.bt_stopped_after_anomaly = False
        # נחדש את הלוגים של BT
        self.root.after(300, self.schedule_bt_log)


def main():
    root = tk.Tk()
    gui = LogGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
