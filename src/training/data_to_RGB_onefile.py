import os
import openpyexcel
import shutil
from PIL import Image
import numpy as np
from skimage import exposure
row_size = 3
# column_size =16725
# 129 x129
column_size =16641
columns = 0

original_dataset_dir = "E:/USB/논문/논문/S_SCI_BLDC_HALL_SENSOR_AI_IEEE_ACCESS/데이터/0.0001_129"
download_dataset_dir ="E:/USB/논문/논문/S_SCI_BLDC_HALL_SENSOR_AI_IEEE_ACCESS/데이터/raw_files"

# shutil.rmtree(original_dataset_dir, ignore_errors=True)
# os.mkdir(original_dataset_dir)
excel = 'Hall_C0.xlsx'
# excel = 'NO_DELAY.xlsx'
# fnames = ['no_delay.{}.png'.format(i) for i in range(20)]
fnames = ['hall_C0_current.{}.png'.format(i) for i in range(20)]

for fname in fnames:
 kk = os.path.join(download_dataset_dir,excel)
 wb=openpyexcel.load_workbook(kk)
 sh=wb['Sheet1']
 arr = [[0 for _ in range(column_size)] for _ in range(row_size)]
 for i in range(row_size):
  for j in range(column_size):
   arr[i][j]= sh.cell(j+columns+1+200000,i+1).value
 arr_np = np.array(arr, dtype=np.float64)
 print (arr_np)
 flatten=arr_np.flatten()
 #print(flatten.shape)
 # reshaped= flatten.reshape(225, 223)*10000
 reshaped = flatten.reshape(129, 129, 3) * 10000
 print(reshaped)
 # normalization
 normalized_image = exposure.rescale_intensity(reshaped, in_range='image', out_range=(0, 255))
 print(normalized_image)
 # Create an Image object from the NumPy array
 image = Image.fromarray(normalized_image, 'RGB')
 image_array = np.array(image)
 print (image_array.shape)
 # Resize the image to 224x224
 # image = image.resize((224, 224))
 print(image.size)
 s = os.path.join(original_dataset_dir, fname)
 print(s)
 image.save(s)
image.show()
