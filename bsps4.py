import cv2
import pytesseract
from PIL import ImageGrab
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

screen_region = (166, 76, 368, 957)  # Example: (left, top, width, height)

i = 0

while i < 10:
    screenshot = ImageGrab.grab(bbox=screen_region)

    screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    text = pytesseract.image_to_string(screenshot_np)

    print("Extracted Text:", text)

    # Display the screenshot
    cv2.imshow('Screenshot', screenshot_np)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    i += 1

cv2.destroyAllWindows()
