"""
Sample code for load the 8000 pre-processed background image data.
Before running, first download the files from:
  https://github.com/ankush-me/SynthText#pre-generated-dataset
"""

from email.mime import base
import h5py
import numpy as np
from PIL import Image
import os.path as osp
import pickle as cp
from synthgen import *
from common import *
import os


base_dir = '../SynthOrig/'
im_dir = os.path.join(base_dir, 'bg_img')
depth_db = h5py.File( os.path.join(base_dir,'depth.h5'),'r')
seg_db = h5py.File(os.path.join(base_dir, 'seg.h5'),'r')
imnames_cp = os.path.join(base_dir, 'imnames.cp')

LANG ='EN'
OUT_FILE = f'results/Synth{LANG}.h5'

NUM_IMG = -1
SECS_PER_IMG = 5
INSTANCE_PER_IMAGE = 1 
DATA_PATH = f'support_{LANG.lower()}'

def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']        
    text_utf8 = [char.encode('utf8') for char in res[i]['txt']]
    db['data'][dname].attrs['txt'] = text_utf8



imnames = sorted(depth_db.keys())

if os.path.exists(imnames_cp):
  with open(imnames_cp, 'rb') as f:
    filtered_imnames = set(cp.load(f))
else:
  filtered_imnames = imnames
  

def main(viz=False, lang='EN'):
    # open the output h5 file:
    out_db = h5py.File(OUT_FILE,'w')
    out_db.create_group('/data')

    RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG, lang=lang)

    gen_count = 0
    for i,imname in enumerate(imnames):
        try:
            # ignore if not in filetered list:
            if imname not in filtered_imnames: continue
            
            # get the colour image:
            img = Image.open(osp.join(im_dir, imname)).convert('RGB')
            
            # get depth:
            depth = depth_db[imname][:].T
            if len(depth.shape) == 3:
              depth = depth[:,:,0] ## Comment this when one channel available

            # get segmentation info:
            seg = seg_db['mask'][imname][:].astype('float32')
            area = seg_db['mask'][imname].attrs['area']
            label = seg_db['mask'][imname].attrs['label']

            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz,Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))
            
            # see `gen.py` for how to use img, depth, seg, area, label for further processing.
            #    https://github.com/ankush-me/SynthText/blob/master/gen.py

            res = RV3.render_text(img,depth,seg,area,label,ninstance=INSTANCE_PER_IMAGE,viz=viz)
            if len(res) > 0:
                # non-empty : successful in placing text:
                add_res_to_db(imname,res,out_db)
            # visualize the output:
            if viz:
                if 'q' in input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
                    break
            gen_count += 1
            print('Images', gen_count, i+1)
        except:
            traceback.print_exc()
            print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
            continue
    out_db.close()



if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  parser.add_argument('--lang', default='EN',help='Select language : EN/JP')
  args = parser.parse_args()
  main(args.viz, args.lang)