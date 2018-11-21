from PIL import Image
from resizeimage import resizeimage
import os

path = 'Users/alexanderboone/Desktop/test/AllNavPlots_jpg/data/'
for file_name in os.listdir(path):
	if file_name.endswith('.jpg'):
			with Image.open(file_name) as image:
				cover = resizeimage.resize_cover(image, [300, 300])
				cover.save(file_name, image.format)