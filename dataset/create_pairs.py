#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os, cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from random import shuffle, seed
from multiprocessing import Pool, Manager


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Get files
image_files  = sorted(glob("/media/antiaegis/storing/datasets/HumanSeg/EG/data_for_run/images/*.*"))
image_files += sorted(glob("/media/antiaegis/storing/datasets/HumanSeg/Supervisely/data_for_run/images/*.*"))

label_files  = sorted(glob("/media/antiaegis/storing/datasets/HumanSeg/EG/data_for_run/labels/*.*"))
label_files += sorted(glob("/media/antiaegis/storing/datasets/HumanSeg/Supervisely/data_for_run/labels/*.*"))

assert len(image_files)==len(label_files)
n_files = len(image_files)

# Shuffle
seed(0)
shuffle(image_files)
seed(0)
shuffle(label_files)

# Count number of pixels belong to categories
manager = Manager()
foregrounds = manager.list([])
backgrounds = manager.list([])

def pool_func(args):
    label_file = args
    img = cv2.imread(label_file, 0)
    foreground = np.sum((img>0).astype(np.uint8)) / img.size
    background = np.sum((img==0).astype(np.uint8)) / img.size
    foregrounds.append(foreground)
    backgrounds.append(background)

pools = Pool(processes=8)
args = label_files
for _ in tqdm(pools.imap_unordered(pool_func, args), total=len(label_files)):
    pass

foregrounds = [element for element in foregrounds]
backgrounds = [element for element in backgrounds]
print("foregrounds:", sum(foregrounds)/n_files)
print("backgrounds:", sum(backgrounds)/n_files)
print("ratio:", sum(foregrounds) / sum(backgrounds))

# Divide into 3 groups: small, averg, and large
RATIO = [0.2, 0.8]
averg_ind = []
for idx, foreground in enumerate(foregrounds):
    if RATIO[0] <= foreground <= RATIO[1]:
        averg_ind.append(idx)
print("Number of averg indices:", len(averg_ind))

# Split train/valid
RATIO = 0.9
TRAIN_FILE = "dataset/antiaegis_train_mask.txt"
VALID_FILE = "dataset/antiaegis_valid_mask.txt"

shuffle(averg_ind)
ind_train = averg_ind[:int(RATIO*len(averg_ind))]
ind_valid = averg_ind[int(RATIO*len(averg_ind)):]
print("Number of training samples:", len(ind_train))
print("Number of validating samples:", len(ind_valid))

fp = open(TRAIN_FILE, "w")
for idx in ind_train:
    image_file, label_file = image_files[idx], label_files[idx]
    line = "%s, %s" % (image_file, label_file)
    fp.writelines(line + "\n")

fp = open(VALID_FILE, "w")
for idx in ind_valid:
    image_file, label_file = image_files[idx], label_files[idx]
    line = "%s, %s" % (image_file, label_file)
    fp.writelines(line + "\n")


#------------------------------------------------------------------------------
#   Check training dataset
#------------------------------------------------------------------------------
fp = open(TRAIN_FILE, 'r')
lines = fp.read().split("\n")
lines = [line.strip() for line in lines if len(line)]
lines = [line.split(", ") for line in lines]

print("Checking %d training samples..." % (len(lines)))
for line in lines:
    image_file, label_file = line
    if not os.path.exists(image_file):
        print("%s does not exist!" % (image_file))
    if not os.path.exists(label_file):
        print("%s does not exist!" % (label_file))


#------------------------------------------------------------------------------
#   Check validating dataset
#------------------------------------------------------------------------------
fp = open(VALID_FILE, 'r')
lines = fp.read().split("\n")
lines = [line.strip() for line in lines if len(line)]
lines = [line.split(", ") for line in lines]

print("Checking %d validating samples..." % (len(lines)))
for line in lines:
    image_file, label_file = line
    if not os.path.exists(image_file):
        print("%s does not exist!" % (image_file))
    if not os.path.exists(label_file):
        print("%s does not exist!" % (label_file))