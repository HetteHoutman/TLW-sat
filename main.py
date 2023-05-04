from satpy import Scene
from glob import glob

filenames = glob('data/MSG4*/*.nat')
global_scene = Scene(reader="seviri_l1b_hrit", filenames=filenames)