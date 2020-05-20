import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
import wget, tarfile
import cv2
from PIL import Image
def get_data(DB_FNAME):
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.BLUE,'\tdownloading data (56 M) from: '+DATA_URL,bold=True)
      print
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\tdata saved at:'+DB_FNAME,bold=True)
      sys.stdout.flush()
    except:
      print colorize(Color.RED,'Data not found and have problems downloading.',bold=True)
      sys.stdout.flush()
      sys.exit(-1)
  # open the h5 file and return:
  return h5py.File(DB_FNAME,'r')



def add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path):
  db=h5py.File(DB_FNAME,'w')
  depth_db=get_data(more_depth_path)
  seg_db=get_data(more_seg_path)
  db.create_group('image')
  db.create_group('depth')
  db.create_group('seg')
  for imname in os.listdir(more_img_file_path):
    if imname.endswith('.jpg'):
      full_path=more_img_file_path+imname
      print full_path,imname
      try:
        j=Image.open(full_path)
        imgSize=j.size
        rawData=j.tostring()
        img=Image.fromstring('RGB',imgSize,rawData)
        #img = img.astype('uint16')
      	db['image'].create_dataset(imname,data=img)
      	db['depth'].create_dataset(imname,data=depth_db[imname])
      	db['seg'].create_dataset(imname,data=seg_db['mask'][imname])
      	db['seg'][imname].attrs['area']=seg_db['mask'][imname].attrs['area']
      	db['seg'][imname].attrs['label']=seg_db['mask'][imname].attrs['label']

      except Exception as e:
        print e
  db.close()
  depth_db.close()
  seg_db.close()


# path to the data-file, containing image, depth and segmentation:
DB_FNAME = './dset_8000.h5'

#add more data into the dset
more_depth_path='/data/home/xuhuaren/ft_local/SynthText_Chinese_version-master/data/ft_local/depth.h5'
more_seg_path='/data/home/xuhuaren/ft_local/SynthText_Chinese_version-master/data/ft_local/seg.h5'
more_img_file_path='/data/home/xuhuaren/ft_local/SynthText_Chinese_version-master/data/ft_local/bg_img/'

add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path)
