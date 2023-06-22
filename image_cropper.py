from PIL import Image
import os

names = os.listdir("./Images")

for name in names:
    filename = os.path.join(f"./Images/{name}")
    img = Image.open(filename)
    img = img.resize((1024, 1024))
    img.save(f"./Images/{name}")

names = os.listdir("./Masks")

for name in names:
    filename = os.path.join(f"./Masks/{name}")
    img = Image.open(filename)
    img = img.resize((1024, 1024))
    img.save(f"./Masks/{name}")
