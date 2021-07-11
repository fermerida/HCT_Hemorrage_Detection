import sys
import glob
import h5py
import cv2
import os
IMG_WIDTH = 30
IMG_HEIGHT = 30

h5file = '../dataset/dataset.h5'

dirname = './data/clases/**/*.png'
nfiles = len(glob.glob(dirname))
print(f'count of image files nfiles={nfiles}')

def comprise():
    # resize all images and load into a single dataset
    with h5py.File(h5file,'w') as  h5f:
        img_ds = h5f.create_dataset('images',shape=(nfiles, IMG_WIDTH, IMG_HEIGHT,3), dtype=int)
        for cnt, ifile in enumerate(glob.iglob(dirname)) :
            img = cv2.imread(ifile, cv2.IMREAD_COLOR)
            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            img_resize = cv2.resize( img, (IMG_WIDTH, IMG_HEIGHT) )
            img_ds[cnt:cnt+1:,:,:] = img_resize
    print('New dataset created and added')

if __name__ == "__main__":
    comprise()