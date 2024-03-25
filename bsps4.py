import numpy as np
import cv2
import pyautogui
import time

# take screenshot using pyautogui
start_time = time.time()  # Record start time
i = 0
while i < 1:
    image = pyautogui.screenshot(f"image{i}.png")
    i += 1
end_time = time.time()  # Record end time

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
