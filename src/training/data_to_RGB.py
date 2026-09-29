import os
import openpyexcel
import shutil
from PIL import Image
import numpy as np
row_size = 3
column_size =14700
columns = 0

original_dataset_dir = "C:/Users/82106/PycharmProjects/original/Hall_C"
download_dataset_dir ="E:/USB/논문/논문/S_inverter_Hallsensor_AI_IEEE_ACCESS/"

shutil.rmtree(original_dataset_dir, ignore_errors=True)
os.mkdir(original_dataset_dir)
excels = ['Hall_C{}.xlsx'.format(i) for i in range(2)]
print(excels)
#fnames = ['phase_current.{}.png'.format(i) for i in range(3)]
#print(fnames)
#for excel, fname in zip(excels, fnames):
jump=0
for excel in excels:
 fnames = ['hall_C_current.{}.png'.format(i+jump) for i in range(20)]
 jump=20
 columns=0
 for fname in fnames:
  print(fname)
  kk = os.path.join(download_dataset_dir,excel)
  print(kk)
  wb=openpyexcel.load_workbook(kk)
  sh=wb['Sheet1']
  arr = [[0 for _ in range(column_size)] for _ in range(row_size)]
  for i in range(row_size):
   for j in range(column_size):
     arr[i][j]= sh.cell(j+columns+1+200000,i+1).value
 # Convert the 2D array to a NumPy array
  columns=columns+14700
  arr_np = np.array(arr, dtype=np.float64)
  print (arr_np)
  flatten=arr_np.flatten()
 #print(flatten.shape)
  reshaped= flatten.reshape(210, -1)
  print(reshaped.shape)
  # Create an Image object from the NumPy array
  image = Image.fromarray(reshaped, 'RGB')
  s = os.path.join(original_dataset_dir, fname)
  print(s)
  image.save(s)

# width, height = image.size
#print(height, width)
# Save the image
###############################
#fnames = ['phase_voltage.{}.png'.format(i) for i in range(3)]
#for fname in fnames:
# s = os.path.join(original_dataset_dir,fname)
# image.save(s)
# Display the image
image.show()
