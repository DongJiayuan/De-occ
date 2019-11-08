from PIL import Image
from shutil import copyfile
import os

boxfile = '.\\data\\40_labels.txt'
images_parent_path = '.\\data\\celeb\\cropped_celeba'
occ_path = '.\\data\\celeb\\occlude_celeba_subset'
nonocc_path = '.\\data\\celeb\\non-occlude_celeba_subset'

f = open(boxfile)
f.readline()
line = f.readline()
while line:
    img_info = f.readline()
    info_list = img_info.split(' ')
    filename = info_list[0]
    Glasses = info_list[16]
    Hat = info_list[36]
    print(filename)
    source_file = os.path.join(images_parent_path,filename)
    if not os.path.exists(source_file):
        continue
    if Glasses == '1' or Hat == '1':
        #remove this file to another directory
        copyfile(source_file, os.path.join(occ_path,filename))
    else:
        copyfile(source_file, os.path.join(nonocc_path,filename))