import torch
import cv2
import numpy as np
from PIL.Image import Image


class GrapeFeatureExtractor():
    """
    Extracts histogram features from cropped grape images.
    """

    def __init__(self):
        # FIXME Cuda non credo serva in questo caso
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def extract(self, images, nbin_rows, nbin_cols, cross=None, hsv=False):
        """
        :param images: a single image or a list of images. These should be of type numpy.ndarray, or PIL.Image
        :param nbin_cols: number of bins in the width direction
        :param nbin_rows: number of bins in the height direction
        :param cross: None or tuple. The tuple should have four numbers: (start_row, end_row, start_col, end_col),
        where the row and col values here are referred to the histogram grid, not to the image pixels. Example: ina
        8x8 histogram grid a possible cross is [1,5,1,8] to keep the bins that are in the second to seventh column,
        or in the second to fourth row, included.
        :param hsv: boolean value telling if the image should be converted to hsv before computing the histogram.
        :return:
        """
        # check input values TODO: write tests for image value errors
        if nbin_rows < 1 or nbin_cols < 1:
            raise ValueError(f'nbin_rows and nbin_cols must be grater than 0. Found {nbin_rows}, {nbin_cols}')
        if not isinstance(images, list):
            images = [images]
        for i, im in enumerate(images):
            if not(isinstance(im, Image) or isinstance(im, np.ndarray)):
                raise ValueError(f"Input images should be numpy ndarray or Images. Found {type(im)}")
            # convert all formats to numpy array
            if isinstance(im, Image):
                im = np.array(im)
            if im.dtype == np.uint8:
                im = im.astype(np.single)/255
            images[i] = im
        if nbin_rows > images[0].shape[0] or nbin_cols > images[0].shape[1]:
            raise ValueError("bin number cannot be grater than pixel dimenstion in W for H.\n"
                             f"Image size {images[0].shape}, nbin_rows: {nbin_rows} nbin_cols: {nbin_cols}")
        if cross:
            if not (isinstance(cross, tuple) or isinstance(cross, list)):
                raise ValueError(f"Parameter 'cross' should be an instance of tuple or list types, found {type(cross)}")
            if len(cross) != 4:
                raise ValueError(f"Parameter cross should have 4 values, found {len(cross)}")
            if cross[0] < 0 or cross[1] > nbin_rows or cross[2] < 0 or cross[3] > nbin_cols:
                raise ValueError(f"Parameter cross values should be in the 0-{nbin_rows} and 0-{nbin_cols} ranges, found {cross}")
        hist_batch = list()
        for image in images:
            # convert image to HSV color space
            if hsv:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
                image[:,:,0] = image[:,:,0] / 360
            # normalize to float
            # if image.dtype == np.uint8:
            #     image = image.astype(np.float64)/255
            bin_h = image.shape[0] // nbin_rows
            bin_w = image.shape[1] // nbin_cols
            hist = list()
            for i in range(nbin_rows):
                for j in range(nbin_cols):
                    if cross:
                        if (i < cross[0] or i >= cross[1]) and (j < cross[2] or j >= cross[3]):
                            continue
                    bin = image[i*bin_h:(i+1)*bin_h, j*bin_w:(j+1)*bin_w]
                    avg_clr = np.mean(bin, axis=(0,1))
                    hist.append(avg_clr)
            hist_batch.append(np.array(hist).flatten())

        hist_batch = np.array(hist_batch)
        return hist_batch