#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import cv2
import numpy as np


#------------------------------------------------------------------------------
#   BaseInference
#------------------------------------------------------------------------------
class BaseInference(object):
    def __init__(self, model):
        self.model = model


    def load_image(self, path):
        raise NotImplementedError


    def preprocess(self, image):
        raise NotImplementedError


    def predict(self, X):
        raise NotImplementedError


    def draw_matting(self, image, mask):
        """
        image (np.uint8) shape (H,W,3)
        mask  (np.float32) range from 0 to 1, shape (H,W)
        """
        mask = 255*(1.0-mask)
        mask = np.expand_dims(mask, axis=2)
        mask = np.tile(mask, (1,1,3))
        mask = mask.astype(np.uint8)
        image_alpha = cv2.add(image, mask)
        return image_alpha


    def draw_transperency(self, image, mask, color_f=[255,0,0], color_b=[0,0,255]):
        """
        image (np.uint8) shape (H,W,3)
        mask  (np.float32) range from 0 to 1, shape (H,W)
        """
        mask = mask.round()
        alpha = np.zeros_like(image, dtype=np.uint8)
        alpha[mask==1, :] = color_f
        alpha[mask==0, :] = color_b
        image_alpha = cv2.add(image, alpha)
        return image_alpha


    def draw_background(self, image, mask, background, kernel_sz=25, sigma=0):
        """
        image (np.uint8) shape (H,W,3)
        mask  (np.float32) range from 0 to 1, shape (H,W)
        background (np.uint8) shape (H,W,3)
        """
        image = image.astype(np.float32)
        background = background.astype(np.float32)

        mask_filtered = cv2.GaussianBlur(mask, (kernel_sz, kernel_sz), sigma)
        mask_filtered = np.expand_dims(mask_filtered, axis=2)
        mask_filtered = np.tile(mask_filtered, (1,1,3))

        image_alpha = image*mask_filtered + background*(1-mask_filtered)
        return image_alpha.astype(np.uint8)