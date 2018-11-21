from pdf2image import convert_from_path
import os 

path = '/Users/alexanderboone/Desktop/test/standards/Standard_Indiv_Plots/'
for file_name in os.listdir(path):
  if file_name.endswith('.pdf'):
    pages = convert_from_path(path + file_name, thread_count=4)
    for page in pages:
      page.save(file_name + '.jpg', 'JPEG')