import cv2
import os
from mask import create_mask

folder_path =r"C:\Users\kiwi\Desktop\face-mask-detector\Data_Generator\Downloads"

# folder_path = "/home/preeth/Downloads"
#dist_path = "/home/preeth/Downloads"

#c = 0
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for i in range(len(images)):
    print("the path of the image is", images[i])
    #image = cv2.imread(images[i])
    #c = c + 1
    create_mask(images[i])



