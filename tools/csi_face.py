import serial
import numpy as np

ser = serial.Serial("COM9", 921600)

while True:
    line = ser.readline().decode(errors="ignore")

    if not line.startswith("CSI_DATA"):
        continue

    parts = line.split("[")[1]
    parts = parts.split("]")[0] 
    nums = [int(x) for x in parts.split()]
    arr = np.array(nums)

    imags = arr[0::2]
    reals = arr[1::2]
    magnitude = np.sqrt(imags **2 + reals ** 2)
    print(len(magnitude))