from PIL import Image
import os

names = os.listdir("./Masks")

max_res = 0
full_res = (0, 0)

for name in names:
    filename = os.path.join(f"./Masks/{name}")
    img = Image.open(filename)
    img = img.resize((1024, 1024))
    img.save(f"./Masks/{name}")
