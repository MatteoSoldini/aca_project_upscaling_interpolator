from PIL import Image

img = Image.open("input.jpg")
img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS).save("lanczos.jpg")
img.resize((img.width * 2, img.height * 2), Image.Resampling.BILINEAR).save("bilinear.jpg")
img.resize((img.width * 2, img.height * 2), Image.Resampling.BICUBIC).save("bicubic.jpg")
img.resize((img.width * 2, img.height * 2), Image.Resampling.NEAREST).save("nearest.jpg")
