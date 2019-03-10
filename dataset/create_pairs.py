#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os
from glob import glob
from random import shuffle, seed


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Get files
image_files = sorted(glob("/media/antiaegis/storing/datasets/HumanSeg/EG/data_for_run/images/*.*"))
image_files += sorted(glob("/media/antiaegis/storing/datasets/HumanSeg/Supervisely/data_for_run/images/*.*"))

label_files = sorted(glob("/media/antiaegis/storing/datasets/HumanSeg/EG/data_for_run/labels/*.*"))
label_files += sorted(glob("/media/antiaegis/storing/datasets/HumanSeg/Supervisely/data_for_run/labels/*.*"))

assert len(image_files)==len(label_files)
n_files = len(image_files)

# Shuffle
seed(0)
shuffle(image_files)
seed(0)
shuffle(label_files)

# Split train/valid
RATIO = 0.9
N_TRAIN = int(RATIO * n_files)
print("Number of training samples:", N_TRAIN)
print("Number of validating samples:", n_files-N_TRAIN)

# Train dataset
fp = open("dataset/train_pairs.txt", 'w')
for image, label in zip(image_files[:N_TRAIN], label_files[:N_TRAIN]):
    line = "%s, %s" % (image, label)
    fp.writelines(line + "\n")

# Valid dataset
fp = open("dataset/valid_pairs.txt", 'w')
for image, label in zip(image_files[N_TRAIN:], label_files[N_TRAIN:]):
    line = "%s, %s" % (image, label)
    fp.writelines(line + "\n")


#------------------------------------------------------------------------------
#   Check training dataset
#------------------------------------------------------------------------------
fp = open("dataset/train_pairs.txt", 'r')
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
fp = open("dataset/valid_pairs.txt", 'r')
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