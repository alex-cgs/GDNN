import cv2
import pytesseract
from PIL import ImageGrab
import numpy as np
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

screen_type = "laptop" # "laptop" or "desktop"
screen_region = (166, 76, 368, 957) if screen_type == "desktop" else (77, 106, 296, 1081)

view_region = (1444, 132, 2360, 1295) if screen_type == "desktop" else (1444, 147, 2360, 1438)

# laptop: (77, 106, 296, 1081) # desktop: (166, 76, 368, 957)  # Example: (left, top, width, height)

MAX_TIME = 770 # 12 minutes and 50 seconds
MAX_ATTEMPTS = 10000
MAX_X = 100000
MAX_Y = 5000

def transition():
    try:
        screenshot = ImageGrab.grab(bbox=screen_region)
        screenshot_np = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        text = pytesseract.image_to_string(screenshot_np).split()
        
        t = float(text[3])
        att = float(text[5])
        x = float(text[13])
        y = float(text[15])
        
        return t, att, x, y
    except:
        time.sleep(0.5)
        return transition()

    # Display the screenshot
    # cv2.imshow('Screenshot', screenshot_np)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# transition()

def state(att1, x1, y1, att2, x2, y2):
    return "dead" if att1 == att2 and x1 == x2 and y1 == y2 else "play"


# WIN: (490, 140) == (86, 255, 255), (1153, 418) == (54, 141, 0)
# LOSE: only first one not second
# PLAY: None
def state_from_screen():
    px = ImageGrab.grab().load()
    
    if screen_type == "desktop":
        pixel1 = px[490, 140]
        pixel2 = px[1153, 418]
    else:
        pixel1 = px[435, 194]
        pixel2 = px[1153, 480]
    
    if np.array_equal(pixel1, [86, 255, 235]) and np.array_equal(pixel2, [54, 141, 0]):
        return "WIN"
    elif np.array_equal(pixel1, [86, 255, 235]):
        return "LOSE"
    else:
        return "PLAY"


def get_file_binary(idx):
    f = open("level_db/" + str(idx) + ".txt", mode="rb")
    data = list(f.read())
    f.close()
    
    for i in range(len(data)):
        data[i] = data[i] / 255

    return data

def view():
    screenshot = ImageGrab.grab(bbox=view_region)
    screenshot_pixels = screenshot.load()
    pixels = []

    # Loop through every 10th pixel in both dimensions
    for x in range(0, screenshot.width, 10):
        for y in range(0, screenshot.height, 10):
            pixel = screenshot_pixels[x, y]
            pixels.append(pixel[0] / 255)
            pixels.append(pixel[1] / 255)
            pixels.append(pixel[2] / 255)
    
    return pixels

# # Evaluate the function 10 times and measure the time
# total_time = 0
# for _ in range(10):
#     start_time = time.time()
#     state = view()
#     end_time = time.time()
#     total_time += (end_time - start_time)

# # Calculate the average time taken per evaluation
# average_time = total_time / 10

# print(f"Average time taken per evaluation: {average_time} seconds")
# print(state)

# print(get_file_binary(1))

# stri = ['LevellD:', '1', 'Time:', '19.61', 'Attempt:', '1', 'Taps:', '0', 'TimeWarp:', '1', 'Gravity:', '1', 'X:', '508', 'Y:', '105', 'Active:', '2', 'Gradients:', '0', 'Particles:', '0', '--', 'Audio', '--', 'Songs:', '1', 'SFX:', '0', '--', 'Perf', '--', 'Move:', '0', 'Rotate:', '0', 'Scale:', '0', 'Follow:', '0', '--', 'Area', '--', 'Move:', '0/0', 'Rotate:', '0/0', 'Scale:', '0/0', 'ColOp:', '0/0']
# print(float(stri[13]))

# print(transition())