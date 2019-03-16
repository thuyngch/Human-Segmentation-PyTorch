#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import cv2, torch
import numpy as np
from time import time

from models import UNet
from utils import utils


#------------------------------------------------------------------------------
#	Get boundary
#------------------------------------------------------------------------------
def boundary_region(sidmoid, lower_thres=0.45, upper_thres=0.55):
	output = (sidmoid>=lower_thres) * (sidmoid<=upper_thres)
	return output.astype(np.float32)

def boundary_points(sidmoid, lower_thres=0.45, upper_thres=0.55):
	output = (sidmoid>=lower_thres) * (sidmoid<=upper_thres)
	y, x = np.where(output==True)
	points = list(zip(x.tolist(), y.tolist()))
	return points


#------------------------------------------------------------------------------
#	Parameters
#------------------------------------------------------------------------------
# Read video
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('/media/cpu11777/data/ZaloHumanSeg/videos/video13.mp4')

# Model checkpoint
CHECKPOINT = "UNet_ResNet18.pth"

# Boundary
BOUND_LOWER = 0.45
BOUND_UPPER = 0.55

# Prediction frequency
PRED_FREQ = 10

# Parameters for lucas kanade optical flow
lk_params = dict(
	winSize=(15,15),
	maxLevel=2,
	criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


#------------------------------------------------------------------------------
#	Load Deep-learning model to predict the initial boundary mask
#------------------------------------------------------------------------------
# Create model
model = UNet(
	n_classes=1,
	img_layers=3,
	backbone="ResNet",
	backbone_args={
		"n_layers": 18,
		"filters": 64,
		"input_sz": 224,
		"pretrained": None,
	}
)
# Load trained weights
trained_dict = torch.load(CHECKPOINT, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=True)
model.eval()

# Take first frame and predict initial mask
_, old_frame = cap.read()
# old_frame = cv2.transpose(old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
image = old_frame[...,::-1]
h, w = image.shape[:2]

start_time = time()
X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=224, pad_value=0)
with torch.no_grad():
	mask = model(X)[0,0,...].numpy()
end_time = time()
print("Prediction time: %.3f [s]" % (end_time-start_time))

# Resize mask to the original size
mask = mask[pad_up: pad_up+h_new, pad_left: pad_left+w_new]
mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

# Get boundary points
points = boundary_points(mask, lower_thres=BOUND_LOWER, upper_thres=BOUND_UPPER)
p0 = np.zeros([len(points), 1, 2], np.float32)
for idx, point in enumerate(points):
	p0[idx, 0, :] = point
print(p0)
print(p0.shape)

# # Plot
# from matplotlib import pyplot as plt
# plt.figure()
# plt.subplot(2,2,1); plt.imshow(image); plt.imshow(mask, alpha=0.5); plt.axis('off')
# plt.subplot(2,2,2); plt.imshow(image); plt.imshow(mask.round(), alpha=0.5); plt.axis('off')
# plt.subplot(2,2,3); plt.imshow(image); plt.imshow(boundary_region(mask), alpha=0.5); plt.axis('off')
# plt.show()


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
# Create some random colors
color = np.random.randint(0, 255, (len(points), 3))

# Create a mask image for drawing purposes
frame_idx = 0
while(True):
	# Read frame from video
	frame_idx += 1
	_, frame = cap.read()
	# frame = cv2.transpose(frame)


	# Deep-learning prediction
	if frame_idx%PRED_FREQ==0:
		X, pad_up, pad_left, h_new, w_new = utils.preprocessing(frame, expected_size=224, pad_value=0)
		start_time = time()
		with torch.no_grad():
			mask = model(X)[0,0,...].numpy()
		end_time = time()
		print("Deep-learning time: %.3f [s]" % (end_time-start_time))
		mask = mask[pad_up: pad_up+h_new, pad_left: pad_left+w_new]
		mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
		mask = 1 / (1 + np.exp(-mask))

		points = boundary_points(mask, lower_thres=BOUND_LOWER, upper_thres=BOUND_UPPER)
		p0 = np.zeros([len(points), 1, 2], np.float32)
		for idx, point in enumerate(points):
			p0[idx, 0, :] = point

		# mask = (255*(1.0-mask)).astype(np.uint8)
		# mask = np.expand_dims(mask, axis=2)
		# mask = np.tile(mask, [1,1,3])
		# frame = cv2.add(frame, mask)


	# Lucas-Kanede tracker
	else:
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		start_time = time()
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
		good_new = p1[st==1]
		end_time = time()
		print("n_points: %.4d; Lucas-Kanade time: %.3f [s]" % (len(good_new), end_time-start_time))

		for i, new in enumerate(good_new):
			a, b = new.ravel()
			frame = cv2.circle(frame, (a,b), 5, [255,0,0], -1)
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)


		# Draw result
		cv2.imshow('video', frame)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

cv2.destroyAllWindows()
cap.release()