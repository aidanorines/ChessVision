"""Bluetooth shutter remote: turns each button press into an event the main program can check.

Pipeline step: input (trigger). Used by capture.py now and the live demo later.
The remote sends Volume Up, a media key Windows normally keeps for itself. RegisterHotKey asks Windows
to deliver it to us instead (the volume stops changing while this runs, and no window needs focus).

It runs in a background thread on purpose: a hotkey message goes to the thread that registered it, and
OpenCV's waitKey() empties the main thread's message queue, so in the main thread presses could be lost.
"""
import ctypes
import queue
import threading
from ctypes import wintypes

VK_VOLUME_UP = 0xAF      # Windows virtual-key code for Volume Up
WM_HOTKEY = 0x0312       # message Windows posts when a registered hotkey is pressed
MOD_NOREPEAT = 0x4000    # holding the button counts as one press


def start_remote():
    """Start listening; returns a Queue that gets one item per button press."""
    presses = queue.Queue()
    ready = queue.Queue()

    def listen():
        user32 = ctypes.windll.user32
        ok = user32.RegisterHotKey(None, 1, MOD_NOREPEAT, VK_VOLUME_UP)
        ready.put(ok)
        if not ok:
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:   # blocks until a message arrives
            if msg.message == WM_HOTKEY:
                presses.put(True)

    threading.Thread(target=listen, daemon=True).start()   # daemon: dies with the program, freeing the hotkey
    if not ready.get(timeout=5):
        raise RuntimeError("RegisterHotKey failed: another program may already own Volume Up")
    return presses


if __name__ == "__main__":
    # Self-check: press the remote a few times; Ctrl+C to stop.
    import time
    presses = start_remote()
    print("Listening. Press the remote. Ctrl+C to quit.")
    n = 0
    try:
        while True:
            while not presses.empty():
                presses.get()
                n += 1
                print("remote pressed", n)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
