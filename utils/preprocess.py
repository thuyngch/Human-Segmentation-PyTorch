#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import cv2, torch
import numpy as np


#------------------------------------------------------------------------------
#  Resize image
#------------------------------------------------------------------------------
def resize_image_cv2(image, expected_size, pad_value):
	h, w = image.shape[0], image.shape[1]
	if w>h:
		w_new = int(expected_size)
		h_new = int(h * w_new / w)
		image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

		pad = w_new-h_new
		pad_width = ((0,pad), (0,0), (0,0))
		constant_values=((0,pad_value), (0,0), (0,0))

		image = np.pad(
			image,
			pad_width=pad_width,
			mode="constant",
			constant_values=constant_values,
		)

	elif w<h:
		h_new = int(expected_size)
		w_new = int(w * h_new / h)
		image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

		pad = h_new-w_new
		pad_width = ((0,0), (0,pad), (0,0))
		constant_values=((0,0), (0,pad_value), (0,0))

		image = np.pad(
			image,
			pad_width=pad_width,
			mode="constant",
			constant_values=constant_values,
		)

	else:
		image = cv2.resize(image, (expected_size, expected_size), interpolation=cv2.INTER_LINEAR)

	return image


def resize_image(image, expected_size, pad_value):
	h, w = image.shape[0], image.shape[1]

	if w>h:
		w_new = expected_size
		h_new = int(h * w_new / w)
	elif w<h:
		h_new = expected_size
		w_new = int(w * h_new / h)
	else:
		h_new, w_new = expected_size, expected_size

	image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
	return image


#------------------------------------------------------------------------------
#  Preprocessing
#------------------------------------------------------------------------------
fnc_resize = resize_image_cv2

def preprocessing_pytorch(image, expected_size=224, pad_value=0):
	image = fnc_resize(image, expected_size, pad_value)
	X = np.transpose(image, axes=(2, 0, 1))
	X = np.expand_dims(X, axis=0)
	X = torch.tensor(X, dtype=torch.float32)
	X /= 255.0
	return X

def preprocessing_caffe2(image, expected_size=224, pad_value=0):
	image = fnc_resize(image, expected_size, pad_value)
	X = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
	X /= 255.0
	X = np.expand_dims(X, axis=0)
	return X


#------------------------------------------------------------------------------
#  Histogram matching
#------------------------------------------------------------------------------
def hist_match(source, template):
	"""
	Adjust the pixel values of a grayscale image such that its histogram
	matches that of a target image

	Arguments:
	-----------
		source: np.ndarray
			Image to transform; the histogram is computed over the flattened
			array
		template: np.ndarray
			Template image; can have different dimensions to source
	Returns:
	-----------
		matched: np.ndarray
			The transformed output image
	"""

	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()

	# get the set of unique pixel values and their corresponding indices and
	# counts
	_, bin_idx, s_counts = np.unique(source, return_inverse=True,
											return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)

	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]

	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

	return interp_t_values[bin_idx].reshape(oldshape)


#------------------------------------------------------------------------------
#  Draw image with transperency
#------------------------------------------------------------------------------
def draw_transperency(image, mask, color_f, color_b, alpha):
	alpha = np.zeros_like(image, dtype=np.uint8)
	alpha[mask==1, :] = color_f
	alpha[mask==0, :] = color_b
	image_alpha = cv2.add(image, alpha)
	return image_alpha


#------------------------------------------------------------------------------
#  Draw foreground pasted into background
#------------------------------------------------------------------------------
def draw_fore_to_back(image, mask, background, kernel_sz=13, sigma=0, use_hist_match=True):
	mask_filtered = cv2.GaussianBlur(mask, (kernel_sz, kernel_sz), sigma)
	mask_filtered = np.expand_dims(mask_filtered, axis=2)
	mask_filtered = np.tile(mask_filtered, (1,1,3))
	image = hist_match(image, background) if use_hist_match else image
	image_alpha = image*mask_filtered + background*(1-mask_filtered)
	return image_alpha.astype(np.uint8)