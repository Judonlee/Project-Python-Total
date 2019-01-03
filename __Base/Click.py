import pyautogui
import time

# time.sleep(2)
for _ in range(1000):
    # time.sleep(1)
    x, y = pyautogui.position()
    pyautogui.click(x, y)
