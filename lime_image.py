"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm

import pandas as pd
import random

import time

from . import lime_base
from .wrappers.scikit_image import SegmentationAlgorithm

# import needed libraries for segmentation
from skimage.segmentation import slic
import matplotlib.pyplot as plt

import os


class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

        self.color_free_space = False
        self.use_maximum_weight = True
        self.all_weights_zero = False


        self.val_low = 0.0
        self.val_high = 255.0
        self.free_space_shade = 180

    def get_image_and_mask(self, label):
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        
        exp = self.local_exp[label]

        temp = np.zeros(self.image.shape)

        w_sum = 0.0
        w_s = []
        for f, w in exp:
            w_sum += abs(w)
            w_s.append(abs(w))
        max_w = max(w_s)
        if max_w == 0:
            self.all_weights_zero = True
            temp[self.image == 0] = self.free_space_shade
            temp[self.image != 0] = 0.0
            return temp, exp

        segments_labels = np.unique(self.segments)
        #print('segments_labels = ', segments_labels)
        
        for f, w in exp:
            #print('(f, w): ', (f, w))
            f = segments_labels[f]
            #print('segments_labels[f] = ', f)

            if w < 0.0:
                c = -1
            elif w > 0.0:
                c = 1
            else:
                c = 0
            #print('c = ', c)
            
            # free space
            if f == 0:
                #print('free_space, (f, w) = ', (f, w))
                if self.color_free_space == False:
                    temp[self.segments == f, 0] = self.free_space_shade
                    temp[self.segments == f, 1] = self.free_space_shade
                    temp[self.segments == f, 2] = self.free_space_shade
            # obstacle
            else:
                if self.color_free_space == False:
                    if c == 1:
                        temp[self.segments == f, 0] = 0.0
                        if self.use_maximum_weight == True:
                            temp[self.segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w
                        else:
                            temp[self.segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum 
                        temp[self.segments == f, 2] = 0.0
                    elif c == 0:
                        temp[self.segments == f, 0] = 0.0
                        temp[self.segments == f, 1] = 0.0
                        temp[self.segments == f, 2] = 0.0
                    elif c == -1:
                        if self.use_maximum_weight == True:
                            temp[self.segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w
                        else:
                            temp[self.segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum 
                        temp[self.segments == f, 1] = 0.0
                        temp[self.segments == f, 2] = 0.0
                                        
        return temp, exp

    def get_image_and_mask_evaluation(self, label):
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        
        exp = self.local_exp[label]

        temp = np.zeros(self.image.shape)

        w_sum = 0.0
        w_s = []
        for f, w in exp:
            w_sum += abs(w)
            w_s.append(abs(w))
        max_w = max(w_s)
        if max_w == 0:
            self.all_weights_zero = True
            temp[self.image == 0] = self.free_space_shade
            temp[self.image != 0] = 0.0
            return temp, exp

        segments_labels = np.unique(self.segments)
        #print('segments_labels = ', segments_labels)
        
        for f, w in exp:
            #print('(f, w): ', (f, w))
            f = segments_labels[f-1]
            #print('segments_labels[f] = ', f)

            if w < 0.0:
                c = -1
            elif w > 0.0:
                c = 1
            else:
                c = 0
            #print('c = ', c)
            
            # free space
            if f == 0:
                #print('free_space, (f, w) = ', (f, w))
                if self.color_free_space == False:
                    temp[self.segments == f, 0] = self.free_space_shade
                    temp[self.segments == f, 1] = self.free_space_shade
                    temp[self.segments == f, 2] = self.free_space_shade
            # obstacle
            else:
                if self.color_free_space == False:
                    if c == 1:
                        temp[self.segments == f, 0] = 0.0
                        if self.use_maximum_weight == True:
                            temp[self.segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w
                        else:
                            temp[self.segments == f, 1] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum 
                        temp[self.segments == f, 2] = 0.0
                    elif c == 0:
                        temp[self.segments == f, 0] = 0.0
                        temp[self.segments == f, 1] = 0.0
                        temp[self.segments == f, 2] = 0.0
                    elif c == -1:
                        if self.use_maximum_weight == True:
                            temp[self.segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / max_w
                        else:
                            temp[self.segments == f, 0] = self.val_low + (self.val_high - self.val_low) * abs(w) / w_sum 
                        temp[self.segments == f, 1] = 0.0
                        temp[self.segments == f, 2] = 0.0
                                        
        return temp, exp


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

        self.dirCurr = os.getcwd()
    
    # SINGLE INSTANCE
    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """

        import copy
        # save original (grayscale) image
        image_orig = copy.deepcopy(image)

        #if len(image.shape) == 2:
        #    image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        # my change
        '''
        if segmentation_fn is None:
            print('NONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
            segments = segmentation_fn(image)
        elif segmentation_fn == 'custom_segmentation':
            print('CUSTOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOM')
            #segments = self.sm_slic_custom(image_orig, image)
            segments = self.sm_only_obstacles(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
            #segments = self.sm_only_obstacles_new(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
            #segments = self.sm_semantic(image, x_odom, y_odom)
        else:
            segments = segmentation_fn(image)
        '''

        '''
        # original code
        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color
        '''

        # my change - to return grayscale to classifier_fn
        fudged_image = image_orig.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image_orig, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)
        #distance_metric = 'jaccard' - alternative distance metric
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()
        #print('distance_metric = ', distance_metric)
        #print('distances = ', distances)
        #print('model_regressor = ', model_regressor)

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp, segments

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """

        #print('data_labels starts')

        n_features = np.unique(segments).shape[0]

        '''
        # Original perturbation
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        #print('data: ', data)
        #print('data.shape: ', data.shape)
        '''

        test_all_comb = True

        if test_all_comb == True:
            # My perturbation - test all possible combinations
            num_samples = 2 ** n_features
            import itertools
            lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
            data = np.array(lst).reshape((num_samples, n_features))
            show_free_space = False

            if show_free_space == True:
                data[0, :] = 1
                data[-1, :] = 0 # only if I use my perturbation
            else:
                data = data[int(data.shape[0]/2):, :]
                data[0, 1:] = 1
                data[-1, 1:] = 0 # only if I use my perturbation    
            #print('data = ', data)
            print('data.shape = ', data.shape)
        else:
            # My perturbation - n_features perturbations - only 1 segment active per perturbation
            num_samples = n_features
            lst = [[1]*n_features]
            for i in range(1, num_samples):
                lst.append([1]*n_features)
                lst[i][n_features-i] = 0    
            data = np.array(lst).reshape((num_samples, n_features))
            #print('data = ', data)
            print('data.shape = ', data.shape)
            #to_add = np.array([1]*n_features).reshape(1,n_features)
            #data = np.concatenate((to_add,data))
            #print('data = ', data)
    
        labels = []

        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)

        #print('data_labels ends')

        return data, np.array(labels)

    def sm_slic_custom(self, image, img_rgb):
        print('\nsm_slic_custom')
        # import needed libraries
        from skimage.segmentation import slic
        #from skimage.measure import regionprops
        import matplotlib.pyplot as plt
        import numpy as np
        #import pandas as pd
        from skimage.color import gray2rgb
        import copy
        import time
        
        # show original image
        img = copy.deepcopy(image)

        # Find segments_2
        segments_slic = slic(img_rgb, n_segments=15, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.005, max_size_factor=10, slic_zero=False,
                            start_label=1, mask=None)

        segments = np.zeros(img.shape, np.uint8)

        # make one free space segment
        ctr = 0
        segments[:, :] = ctr
        ctr = ctr + 1

        num_of_obstacles = 0
        # add obstacle segments        
        for i in np.unique(segments_slic):
            temp = img[segments_slic == i]
            count_of_99_s = np.count_nonzero(temp == 99)
            #print('count: ', count)
            #print('temp: ', temp)
            #print('len(temp): ', temp.shape[0])
            if np.all(img[segments_slic == i] == 99) or count_of_99_s > 0.95 * temp.shape[0]:
                #print('obstacle')
                segments[segments_slic == i] = ctr
                ctr = ctr + 1
                num_of_obstacles += 1

        #print('num_of_obstacles: ', num_of_obstacles)
        #print('np.unique(segments): ', np.unique(segments))

        segmentation_plot = True
        if segmentation_plot == True:
            dirName = 'explanation_results'
            try:
                os.mkdir(dirName)
            except FileExistsError:
                pass
            
            big_plot_size = True
            w = h = 1.6
            if big_plot_size == True:
                w *= 3
                h *= 3

            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments_slic.astype('float64'), aspect='auto')
            fig.savefig(self.dirCurr + '/' + dirName + '/segments_slic.png', transparent=False)
            fig.clf()
            
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments.astype('float64'), aspect='auto')
            fig.savefig(self.dirCurr + '/' + dirName + '/segments_final.png', transparent=False)
            fig.clf()
            
        return segments

    def sm_only_obstacles(self, image, img_rgb, x_odom, y_odom, devDistance_x, sign_x, devDistance_y, sign_y, devDistance, plan_x_list, plan_y_list):
        # import needed libraries
        from skimage.segmentation import slic
        #from skimage.measure import regionprops
        import matplotlib.pyplot as plt
        import numpy as np
        #import pandas as pd
        from skimage.color import gray2rgb
        import copy
        import time

        # show original image
        img = copy.deepcopy(image)

        # Find segments_2
        segments_slic = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False,
                            start_label=1, mask=None)

        segments = np.zeros(img.shape, np.uint8)

        '''
        devDistance_x = int(devDistance_x)
        devDistance_x = abs(devDistance_x)
        devDistance_y = int(devDistance_y)
        devDistance_y = abs(devDistance_y)
        devDistance = int(devDistance)
        '''

        '''
        footprint_radius = 13
        add = 15
        '''

        d_x = plan_x_list[-1] - x_odom
        if d_x == 0:
            d_x = 1
        k = (-plan_y_list[-1] + y_odom) / (d_x)

        #print('abs(k) = ', abs(k)) 

        '''
        delta_y = plan_y_list[-1] - y_odom
        delta_x = plan_x_list[-1] - x_odom  
        '''

        # make one free space segment
        ctr = 0
        segments[:, :] = ctr
        ctr = ctr + 1

        num_of_obstacles = 0
        # add obstacle segments        
        for i in np.unique(segments_slic):
            temp = img[segments_slic == i]
            count_of_99_s = np.count_nonzero(temp == 99)
            #print('count: ', count)
            #print('temp: ', temp)
            #print('len(temp): ', temp.shape[0])
            if np.all(img[segments_slic == i] == 99) or count_of_99_s > 0.95 * temp.shape[0]:
                #print('obstacle')
                segments[segments_slic == i] = ctr
                ctr = ctr + 1
                num_of_obstacles += 1

        #print('num_of_obstacles: ', num_of_obstacles)        

        if 8 > num_of_obstacles > 0:
            # divide segment obstacles    
            seg_labels = np.unique(segments)[1:]        

            num_of_seg = len(seg_labels)
            num_of_wanted_seg = 8

            #print('\nnumber of wanted segments: ', num_of_wanted_seg)
            #print('number of current segments: ', num_of_seg)

            seg_sizes = []

            if num_of_seg < num_of_wanted_seg:
                for i in range(0, num_of_seg):
                    #print(len(segments[segments == seg_labels[i]]))
                    seg_sizes.append(len(segments[segments == seg_labels[i]]))

            #print('\nsizes of segments original: ', seg_sizes)
            #print('labels of segments original: ', seg_labels)
            seg_labels = [x for _, x in sorted(zip(seg_sizes, seg_labels))]
            seg_labels.reverse()
            #print('\nsizes of segemnts sorted: ', seg_sizes)
            #print('labels of segemnts sorted: ', seg_labels)

            seg_missing = num_of_wanted_seg - num_of_seg
            print('\nnumber of segments missing: ', seg_missing)

            # if a number of missing segments is smaller or equal than the number of existing segments
            if seg_missing <= num_of_seg:
                label_current = len(seg_labels) + 1
                for i in range(0, seg_missing):
                    temp = segments[segments == seg_labels[i]]
                    #print('temp = ', temp)

                    # check obstacle shape
                    w_min = 161
                    w_max = -1
                    h_min = 161
                    h_max = -1
                    for j in range(0, segments.shape[0]):
                        for q in range(0, segments.shape[1]):
                            if segments[j, q] == seg_labels[i]:
                                if j > h_max:
                                    h_max = j
                                if j < h_min:
                                    h_min = j
                                if q > w_max:
                                    w_max = q
                                if q < w_min:
                                    w_min = q           

                    #print('\n(h_min, h_max): ', (h_min, h_max))
                    #print('(w_min, w_max): ', (w_min, w_max))

                    height = h_max - h_min + 1
                    #print('\nheight', height)
                    width = w_max - w_min + 1
                    #print('width', width)
           

                    # if upright
                    if abs(k) >= 1:
                        if height > width:
                            for j in range(0, int(len(temp) / 2)):
                                temp[j] = label_current
                            segments[segments == seg_labels[i]] = temp
                        else:
                            label_original = temp[0]
                            num_of_pixels = len(temp)
                            counter = 0
                            finished = False
                            for q in range(0, segments.shape[1]):
                                if finished == True:
                                    break
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        segments[j, q] = label_current
                                        counter += 1
                                        if counter == int(num_of_pixels / 2 + 0.5):
                                            label_current += 1
                                            finished = True
                                            break        

                    # if to the side
                    elif abs(k) < 1:
                        #print('OVAJ SLUCAJ')
                        if width > height:
                            #print('width > height')
                            #print('label_current: ', label_current)
                            label_current
                            for j in range(0, int(len(temp) / 2)):
                                temp[j] = label_current
                            segments[segments == seg_labels[i]] = temp   
                        else:
                            #print('width < height')
                            #print('label_current: ', label_current)
                            label_original = temp[0]
                            #print('label_original = ', label_original)
                            num_of_pixels = len(temp)
                            counter = 0
                            for q in range(0, segments.shape[1]):
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        #print('IN')
                                        #print('counter = ', counter)
                                        if 0 <= counter <= num_of_pixels / 2:
                                            segments[j, q] = label_current
                                        counter += 1

                            '''
                            for j in range(0, height):
                                for q in range(0, int(width/2)):
                                    if j * width + q > len(temp) - 1:
                                        continue
                                    temp[j * width + q] = label_original
                                for q in range(int(width/2), width):
                                    if j * width + q > len(temp) - 1:
                                        continue
                                    temp[j * width + q] = label_current
                            '''
                    
                    label_current += 1

            # if a number of missing segments is greater than the number of existing segments
            else:
                num_of_new_seg_per_old_seg = int(seg_missing / num_of_seg)
                
                # if a number of new segment per old segments is integer and same for all old segments
                if num_of_new_seg_per_old_seg == seg_missing / num_of_seg:
                    #print('CIO BROJ')
                    #print('\nnumber of new segments per existing segment: ', num_of_new_seg_per_old_seg)

                    label_current = len(seg_labels) + 1
                    
                    for i in range(0, num_of_seg):
                        temp = segments[segments == seg_labels[i]]

                        # check obstacle shape
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for j in range(0, segments.shape[0]):
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == seg_labels[i]:
                                    if j > h_max:
                                        h_max = j
                                    if j < h_min:
                                        h_min = j
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q

                        #print('\n(h_min, h_max): ', (h_min, h_max))
                        #print('(w_min, w_max): ', (w_min, w_max))

                        height = h_max - h_min + 1
                        #print('\nheight', height)
                        width = w_max - w_min + 1
                        #print('width', width)

                        # if upright
                        if abs(k) >= 1:
                            #print('UPRIGHT')
                            if height > width:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == seg_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    if counter + 1 < num_of_pixels - num_of_new_seg_per_old_seg:
                                                        label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break
                        # if to the side
                        else:
                            #print('UPRIGHT')
                            if width > height:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == seg_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

                # if a number of new segment per old segments is integer and same for all old segments
                else:
                    #print('NON-CIO BROJ')

                    whole_part = int(seg_missing / num_of_seg)
                    #print('whole part = ', whole_part)
                    rest = seg_missing % num_of_seg
                    #print('rest = ', rest)

                    put_rest_to_biggest_segment = False

                    num_of_new_seg_per_old_seg_list = [whole_part] * num_of_seg

                    if put_rest_to_biggest_segment == False:
                        for i in range(0, rest):
                            num_of_new_seg_per_old_seg_list[i] += 1
                        #print('num_of_new_seg_per_old_seg_list: ', num_of_new_seg_per_old_seg_list)    
                    else:
                        num_of_new_seg_per_old_seg_list[0] += rest
                        #print('num_of_new_seg_per_old_seg_list: ', num_of_new_seg_per_old_seg_list)


                    label_current = len(seg_labels) + 1
                    #print('\nlabel_current = ', label_current)
                    
                    for i in range(0, num_of_seg):
                        temp = segments[segments == seg_labels[i]]

                        # check obstacle shape
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for j in range(0, segments.shape[0]):
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == seg_labels[i]:
                                    if j > h_max:
                                        h_max = j
                                    if j < h_min:
                                        h_min = j
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q

                        #print('\n(h_min, h_max): ', (h_min, h_max))
                        #print('(w_min, w_max): ', (w_min, w_max))

                        height = h_max - h_min + 1
                        #print('\nheight', height)
                        width = w_max - w_min + 1
                        #print('width', width)

                        # if upright
                        if abs(k) >= 1:
                            #print('UPRIGHT')
                            if height > width:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg_list[i] + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == seg_labels[i]] = temp
                            else:
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if counter + 1 < num_of_pixels - num_of_new_seg_per_old_seg_list[i]:
                                                        label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

                        # if to the side
                        else:
                            #print('SIDE')
                            if width > height:
                                #print('WIDTH')
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 1) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg_list[i] + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    #print('label_current: ', label_current)
                                    label_current += 1
                                segments[segments == seg_labels[i]] = temp
                            else:
                                #print('HEIGHT')
                                #print('len(temp): ', len(temp))
                                step = int(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5)
                                #print('step: ', step)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    #print('counter + 1 = ', counter + 1)
                                                    #print('label_current = ', label_current)
                                                    if counter + 1 < num_of_pixels - num_of_new_seg_per_old_seg_list[i]:
                                                        label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                #print('counter_end = ', counter)
                                                label_current += 1
                                                finished = True
                                                break

        segmentation_plot = True
        if segmentation_plot == True:
            dirName = 'explanation_results'
            try:
                os.mkdir(dirName)
            except FileExistsError:
                pass
            
            big_plot_size = True
            w = h = 1.6
            if big_plot_size == True:
                w *= 3
                h *= 3

            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments_slic.astype('float64'), aspect='auto')
            fig.savefig(self.dirCurr + '/' + dirName + '/segments_slic.png', transparent=False)
            fig.clf()
            
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments.astype('float64'), aspect='auto')
            fig.savefig(self.dirCurr + '/' + dirName + '/segments_final.png', transparent=False)
            fig.clf()
            
        # fix labels of segments
        seg_labels = np.unique(segments)
        for i in range(1, len(seg_labels)):
            label = seg_labels[i]
            if label != i:
                segments[segments == label] = i

        #print('\nnp.unique(segments): ', np.unique(segments))
        #print('\nlen(np.unique(segments)): ', len(np.unique(segments)))

        if len(np.unique(segments)) > 9:
            # make one free space segment
            ctr = 0
            segments[:, :] = ctr
            ctr = ctr + 1

            num_of_obstacles = 0
            # add obstacle segments        
            for i in np.unique(segments_slic):
                if np.all(img[segments_slic == i] == 99):
                    #print('obstacle')
                    segments[segments_slic == i] = ctr
                    ctr = ctr + 1
                    num_of_obstacles += 1

            dirName = 'explanation_results'
            try:
                os.mkdir(dirName)
            except FileExistsError:
                pass
            
            big_plot_size = True
            w = h = 1.6
            if big_plot_size == True:
                w *= 3
                h *= 3
            
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments.astype('float64'), aspect='auto')
            fig.savefig(self.dirCurr + '/' + dirName + '/segments_final_corrected.png', transparent=False)
            fig.clf()
            
        return segments

    def sm_only_obstacles_new(self, image, img_rgb, x_odom, y_odom, devDistance_x, sign_x, devDistance_y, sign_y, devDistance, plan_x_list, plan_y_list):
        print('\nsm_only_obstacles started')
        
        # show original image
        #img = copy.deepcopy(image)
        img = image

        sm_only_obstacles_start = time.time()

        # Find segments_2
        segments_slic = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                            multichannel=True, convert2lab=True,
                            enforce_connectivity=True, min_size_factor=0.01, max_size_factor=10, slic_zero=False,
                            start_label=1, mask=None)


        segments = np.zeros(img.shape, np.uint8)

        d_x = plan_x_list[-1] - x_odom
        if d_x == 0:
            d_x = 1
        k = (-plan_y_list[-1] + y_odom) / (d_x)
        #print('\nabs(k) = ', abs(k)) 

        # make one free space segment
        ctr = 0
        segments[:, :] = ctr
        ctr = ctr + 1

        num_of_wanted_obstacle_segments = 8

        num_of_existing_obstacle_segments = 0
        # add existing obstacle segments        
        for i in np.unique(segments_slic):
            temp = img[segments_slic == i]
            count_of_99_s = np.count_nonzero(temp == 99)
            if np.all(img[segments_slic == i] == 99) or count_of_99_s > 0.9 * temp.shape[0]:
                segments[segments_slic == i] = ctr
                ctr = ctr + 1
                num_of_existing_obstacle_segments += 1
        #print('\nnum_of_existing_obstacle_segments: ', num_of_existing_obstacle_segments)
        
        num_of_wanted_obstacle_segments = max(num_of_wanted_obstacle_segments, num_of_existing_obstacle_segments)        
        #print('\nnum_of_wanted_obstacle_segments: ', num_of_wanted_obstacle_segments)
        #num_of_wanted_obstacle_segments = 2 * num_of_existing_obstacle_segments
        
        if num_of_wanted_obstacle_segments > num_of_existing_obstacle_segments > 0:
            #print('IN!!!!!!!!!!!')
            # divide segment obstacles    
            current_segs_labels = np.unique(segments)[1:]        

            #print('number of wanted segments: ', num_of_wanted_obstacle_segments)
            #print('number of current segments: ', num_of_existing_obstacle_segments)
            #print('current_segs_labels = ', current_segs_labels)

            # get current segments sizes
            current_segs_sizes = []
            for i in range(0, num_of_existing_obstacle_segments):
                #print(len(segments[segments == seg_labels[i]]))
                current_segs_sizes.append(len(segments[segments == current_segs_labels[i]]))
            #print('sizes of original segments: ', current_segs_sizes)
            #print('labels of original segments: ', current_segs_labels)
            
            sorted_segs_labels = [x for _, x in sorted(zip(current_segs_sizes, current_segs_labels))]
            sorted_segs_labels.reverse()
            #print('labels of sorted segments: ', sorted_segs_labels)

            num_segs_missing = num_of_wanted_obstacle_segments - num_of_existing_obstacle_segments
            #print('number of segments missing: ', num_segs_missing)

            # if a number of missing segments is smaller or equal than the number of existing segments
            # only divide existing segments into 2
            if num_segs_missing <= num_of_existing_obstacle_segments:
                label_current = len(sorted_segs_labels) + 1
                # for loop from zero until the number of missing segments   
                for i in range(0, num_segs_missing):
                    temp = segments[segments == sorted_segs_labels[i]]
                    #print('\ntemp = ', temp)

                    '''
                    # check obstacle shape
                    w_min = 161
                    w_max = -1
                    h_min = 161
                    h_max = -1
                    for j in range(0, segments.shape[0]):
                        for q in range(0, segments.shape[1]):
                            if segments[j, q] == sorted_segs_labels[i]:
                                if j > h_max:
                                    h_max = j
                                if j < h_min:
                                    h_min = j
                                if q > w_max:
                                    w_max = q
                                if q < w_min:
                                    w_min = q           
                    print('\n(h_min, h_max): ', (h_min, h_max))
                    print('(w_min, w_max): ', (w_min, w_max))
                    height = h_max - h_min + 1
                    print('\nheight = ', height)
                    width = w_max - w_min + 1
                    print('width = ', width)
                    '''

                    #'''
                    # check obstacle shape
                    widths = []
                    heights = []
                    for j in range(0, segments.shape[0]):
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for q in range(0, segments.shape[1]):
                            if segments[j, q] == sorted_segs_labels[i]:
                                if q > w_max:
                                    w_max = q
                                if q < w_min:
                                    w_min = q
                            if segments[q, j] == sorted_segs_labels[i]:
                                if q > h_max:
                                    h_max = q
                                if q < h_min:
                                    h_min = q
                        if w_max >= w_min:
                            widths.append(w_max - w_min + 1)
                        if h_max >= h_min:
                            heights.append(h_max - h_min + 1)
                    height = sum(heights) / len(heights)
                    #print('\nheight = ', height)
                    width = sum(widths) / len(widths)
                    #print('width = ', width)
                    #'''

                    # if upright
                    if abs(k) >= 1:
                        # divide by height
                        if height >= 2 * width:
                            for j in range(0, int(len(temp) / 2)):
                                temp[j] = label_current
                            segments[segments == sorted_segs_labels[i]] = temp
                        
                        # divide by width    
                        else:
                            label_original = temp[0]
                            num_of_pixels = len(temp)
                            counter = 0
                            for q in range(0, segments.shape[1]):
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        if 0 <= counter <= num_of_pixels / 2:
                                            segments[j, q] = label_current
                                        counter += 1

                            '''
                            # second method            
                            label_original = temp[0]
                            num_of_pixels = len(temp)
                            counter = 0
                            finished = False
                            for q in range(0, segments.shape[1]):
                                if finished == True:
                                    break
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        segments[j, q] = label_current
                                        counter += 1
                                        if counter == int(num_of_pixels / 2 + 0.5):
                                            #label_current += 1
                                            finished = True
                                            break
                            '''

                    # if to the side
                    elif abs(k) < 1:
                        # divide by height
                        if width >= 2 * height:
                            for j in range(0, int(len(temp) / 2)):
                                temp[j] = label_current
                            segments[segments == sorted_segs_labels[i]] = temp
                        
                        # divide by width   
                        else:
                            label_original = temp[0]
                            num_of_pixels = len(temp)
                            counter = 0
                            for q in range(0, segments.shape[1]):
                                for j in range(0, segments.shape[0]):
                                    if segments[j, q] == label_original:
                                        if 0 <= counter <= num_of_pixels / 2:
                                            segments[j, q] = label_current
                                        counter += 1
                    
                    label_current += 1

            # if a number of missing segments is greater than the number of existing segments
            else:
                num_of_new_seg_per_old_seg = int(num_segs_missing / num_of_existing_obstacle_segments)
                #print('num_of_new_seg_per_old_seg = ', num_of_new_seg_per_old_seg)
                
                # if a number of new segment per old segments is integer and same for all old segments
                # divide segments in num_of_new_seg_per_old_seg new segments
                if num_of_new_seg_per_old_seg == num_segs_missing / num_of_existing_obstacle_segments:
                    #print('\nnumber of new segments per existing segment: ', num_of_new_seg_per_old_seg)

                    label_current = len(sorted_segs_labels) + 1
                    
                    for i in range(0, num_of_existing_obstacle_segments):
                        temp = segments[segments == sorted_segs_labels[i]]

                        '''
                        # check obstacle shape
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for j in range(0, segments.shape[0]):
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == sorted_segs_labels[i]:
                                    if j > h_max:
                                        h_max = j
                                    if j < h_min:
                                        h_min = j
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q
                        #print('\n(h_min, h_max): ', (h_min, h_max))
                        #print('(w_min, w_max): ', (w_min, w_max))
                        height = h_max - h_min + 1
                        #print('\nheight', height)
                        width = w_max - w_min + 1
                        #print('width', width)
                        '''

                        #'''
                        # check obstacle shape
                        widths = []
                        heights = []
                        for j in range(0, segments.shape[0]):
                            w_min = 161
                            w_max = -1
                            h_min = 161
                            h_max = -1
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == sorted_segs_labels[i]:
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q
                                if segments[q, j] == sorted_segs_labels[i]:
                                    if q > h_max:
                                        h_max = q
                                    if q < h_min:
                                        h_min = q
                            if w_max >= w_min:
                                widths.append(w_max - w_min + 1)
                            if h_max >= h_min:
                                heights.append(h_max - h_min + 1)
                        height = sum(heights) / len(heights)
                        #print('\nheight = ', height)
                        width = sum(widths) / len(widths)
                        #print('width = ', width)
                        #'''

                        # if upright
                        if abs(k) >= 1:
                            # divide by height
                            if height >= 2 * width:
                                step = round(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == sorted_segs_labels[i]] = temp
                            else:
                                step = round(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    if counter + 1 < num_of_pixels - num_of_new_seg_per_old_seg:
                                                        label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break
                        # if to the side
                        else:
                            if width >= 2 * height:
                                step = round(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == sorted_segs_labels[i]] = temp
                            else:
                                step = round(len(temp) / (num_of_new_seg_per_old_seg + 1) + 0.5)
                                label_original = temp[0]
                                num_of_pixels = len(temp)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

                # if a number of new segment per old segments is not integer and not same for all old segments
                else:
                    #print('NON-CIO BROJ')
                    whole_part = int(num_segs_missing / num_of_existing_obstacle_segments)
                    #print('whole part = ', whole_part)
                    rest = num_segs_missing % num_of_existing_obstacle_segments
                    #print('rest = ', rest)

                    put_rest_to_biggest_segment = True

                    num_of_new_seg_per_old_seg_list = [whole_part] * num_of_existing_obstacle_segments

                    if put_rest_to_biggest_segment == False:
                        for i in range(0, rest):
                            num_of_new_seg_per_old_seg_list[i] += 1
                        #print('num_of_new_seg_per_old_seg_list: ', num_of_new_seg_per_old_seg_list)    
                    else:
                        num_of_new_seg_per_old_seg_list[0] += rest
                        #print('num_of_new_seg_per_old_seg_list: ', num_of_new_seg_per_old_seg_list)

                    #print('num_of_new_seg_per_old_seg_list = ', num_of_new_seg_per_old_seg_list)

                    label_current = len(sorted_segs_labels) + 1
                    #print('\nlabel_current = ', label_current)
                    
                    for i in range(0, num_of_existing_obstacle_segments):
                        temp = segments[segments == sorted_segs_labels[i]]

                        '''
                        # check obstacle shape
                        w_min = 161
                        w_max = -1
                        h_min = 161
                        h_max = -1
                        for j in range(0, segments.shape[0]):
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == sorted_segs_labels[i]:
                                    if j > h_max:
                                        h_max = j
                                    if j < h_min:
                                        h_min = j
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q
                        #print('\n(h_min, h_max): ', (h_min, h_max))
                        #print('(w_min, w_max): ', (w_min, w_max))
                        height = h_max - h_min + 1
                        #print('\nheight', height)
                        width = w_max - w_min + 1
                        #print('width', width)
                        '''

                        #'''
                        # check obstacle shape
                        widths = []
                        heights = []
                        for j in range(0, segments.shape[0]):
                            w_min = 161
                            w_max = -1
                            h_min = 161
                            h_max = -1
                            for q in range(0, segments.shape[1]):
                                if segments[j, q] == sorted_segs_labels[i]:
                                    if q > w_max:
                                        w_max = q
                                    if q < w_min:
                                        w_min = q
                                if segments[q, j] == sorted_segs_labels[i]:
                                    if q > h_max:
                                        h_max = q
                                    if q < h_min:
                                        h_min = q
                            if w_max >= w_min:
                                widths.append(w_max - w_min + 1)
                            if h_max >= h_min:
                                heights.append(h_max - h_min + 1)
                        height = sum(heights) / len(heights)
                        #print('\nheight = ', height)
                        width = sum(widths) / len(widths)
                        #print('width = ', width)
                        #'''

                        # if upright
                        if abs(k) >= 1:
                            #print('UPRIGHT')
                            if height >= 2 * width:
                                #print('height > width')
                                step = round(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg_list[i] + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == sorted_segs_labels[i]] = temp
                            else:
                                #print('height <= width')
                                step = round(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5)
                                #print('step = ', step)
                                label_original = temp[0]
                                #print('label_original = ', label_original)
                                num_of_pixels = len(temp)
                                #print('num_of_pixels = ', num_of_pixels)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    label_current += 1
                                            counter += 1
                                            if counter == num_of_pixels:
                                                label_current += 1
                                                finished = True
                                                break

                        # if to the side
                        else:
                            #print('\nSIDE')
                            if width >= 2 * height:
                                #print('\nwidth>height')
                                step = round(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5) # or (... + 0.5) with fixing values from behind
                                for j in range(1, num_of_new_seg_per_old_seg_list[i] + 1):
                                    temp[j*step:(j+1)*step] = label_current
                                    label_current += 1
                                segments[segments == sorted_segs_labels[i]] = temp
                            else:
                                #print('\nwidth<=height')
                                step = round(len(temp) / (num_of_new_seg_per_old_seg_list[i] + 1) + 0.5)
                                #print('\nstep = ', step)
                                label_original = temp[0]
                                #print('\nlabel_original = ', label_original)
                                num_of_pixels = len(temp)
                                #print('\nnum_of_pixels = ', num_of_pixels)
                                counter = 0
                                finished = False
                                for q in range(0, segments.shape[1]):
                                    if finished == True:
                                        break
                                    for j in range(0, segments.shape[0]):
                                        if segments[j, q] == label_original:
                                            if counter < step:
                                                segments[j, q] = label_original
                                            else:
                                                segments[j, q] = label_current
                                                if (counter + 1) % step == 0:
                                                    #print('\ncounter = ', counter)
                                                    #print('step = ', step)
                                                    #print('label_current = ', label_current)
                                                    label_current += 1
                                                    #print('label_current = ', label_current)
                                            counter += 1
                                            if counter == num_of_pixels:
                                                #label_current += 1
                                                finished = True
                                                #break
                                label_current += 1

        segmentation_plot = True
        if segmentation_plot == True:
            dirName = 'explanation_results'
            try:
                os.mkdir(dirName)
            except FileExistsError:
                pass
            
            big_plot_size = True
            w = h = 1.6
            if big_plot_size == True:
                w *= 3
                h *= 3

            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments_slic.astype('float64'), aspect='auto')
            fig.savefig(self.dirCurr + '/' + dirName + '/segments_slic.png', transparent=False)
            fig.clf()

            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments.astype('float64'), aspect='auto')
            fig.savefig(self.dirCurr + '/' + dirName + '/segments_final.png', transparent=False)
            fig.clf()
            
        # fix labels of segments
        seg_labels = np.unique(segments)
        for i in range(0, len(seg_labels)):
            label = seg_labels[i]
            if label != i:
                segments[segments == label] = i

        #print('\nnp.unique(segments): ', np.unique(segments))
        #print('\nlen(np.unique(segments)): ', len(np.unique(segments)))

        sm_only_obstacles_end = time.time()
        sm_only_obstacles_time = sm_only_obstacles_end - sm_only_obstacles_start
        print('\nsm_only_obstacles runtime: ', sm_only_obstacles_time)

        print('\nsm_only_obstacles ended')

        print('\nnp.unique(segments): ', np.unique(segments))

        # if there are no any obstacles, then divide free space in num_of_wanted_obstacle_segments parts
        if len(np.unique(segments)) == 1:
            step = round(segments.shape[0] / (num_of_wanted_obstacle_segments + 1))
            for i in range(0, num_of_wanted_obstacle_segments + 1):
                segments[i*step:(i+1)*step,:] = i

        print('\nnp.unique(segments): ', np.unique(segments))
        
        if len(np.unique(segments)) != num_of_wanted_obstacle_segments + 1:
            print('ERROR!!!')
            return segments_slic

        return segments

    def sm_semantic(self, image, x_odom, y_odom):
        # QSR - qualitative spatial reasoning
        relatum = copy.deepcopy([x_odom, y_odom])
        referent = copy.deepcopy(relatum)
        origin = copy.deepcopy(relatum)
        #print('\norigin = ', origin)
        
        segments = np.zeros(image.shape[:-1], np.uint8)
        height = image.shape[0]
        width = image.shape[1]
        
        semantic_choice = 3

        if semantic_choice == 0:
            # left -- right dichotomy in a relative refence system
            # from 'moratz2008qualitative'
            ctr = 0
            segments[:, :int(relatum[1])] = ctr
            ctr = ctr + 1
            segments[:, int(relatum[1]):] = ctr

            # used for getting semantic costmap
            tpcc_dict = {
                'left': 0,
                'right': 1
            }

            # used for deriving NLP annotations
            tpcc_dict_inv = {v: k for k, v in tpcc_dict.items()}
                
            # reasoning for one random position
            referent = [random.randint(0, width-1), random.randint(0, height-1)]
            print('\nreferent = ', referent)
            print('segments[referent] = ', segments[referent[1],referent[0]])
            if referent == relatum == origin:
                print('origin, relatum (tri, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif origin == relatum:
                print('origin, relatum (dou, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif referent == relatum:
                print('origin, relatum (sam, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            else:
                print('origin, relatum ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ' referent')

        elif semantic_choice == 1:
            # single cross calculus from 'moratz2008qualitative'
            ctr = 0
            segments[:int(relatum[0]), :int(relatum[1])] = ctr
            ctr = ctr + 1
            segments[:int(relatum[0]), int(relatum[1]):] = ctr
            ctr = ctr + 1
            segments[int(relatum[0]):, :int(relatum[1])] = ctr
            ctr = ctr + 1
            segments[int(relatum[0]):, int(relatum[1]):] = ctr

            # used for getting semantic costmap
            tpcc_dict = {
                'left/front': 0,
                'right/front': 1,
                'left/back': 2,
                'right/back': 3
            }

            # used for deriving NLP annotations
            tpcc_dict_inv = {v: k for k, v in tpcc_dict.items()}
                
            # reasoning for one random position
            referent = [random.randint(0, width-1), random.randint(0, height-1)]
            print('\nreferent = ', referent)
            print('segments[referent] = ', segments[referent[1],referent[0]])
            if referent == relatum == origin:
                print('origin, relatum (tri, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif origin == relatum:
                print('origin, relatum (dou, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif referent == relatum:
                print('origin, relatum (sam, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            else:
                print('origin, relatum ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ' referent')

        elif semantic_choice == 2:
            # TPCC reference system
            # my modified version from 'moratz2008qualitative'
            #origin[0] = int(x_odom / 2)
            r = relatum[0] - origin[0]
            
            if r > 0:
                # used for getting semantic costmap
                tpcc_dict = {
                    'csb': 0,
                    'dsb': 1,
                    'clb': 2,
                    'dlb': 3,
                    'cbl': 4,
                    'dbl': 5,
                    'csl': 6,
                    'dsl': 7,
                    'cfl': 8,
                    'dfl': 9,
                    'clf': 10,
                    'dlf': 11,
                    'csf': 12,
                    'dsf': 13,
                    'crf': 14,
                    'drf': 15,
                    'cfr': 16,
                    'dfr': 17,
                    'csr': 18,
                    'dsr': 19,
                    'cbr': 20,
                    'dbr': 21,
                    'crb': 22,
                    'drb': 23
                }
            elif r == 0:
                # used for getting semantic costmap
                tpcc_dict = {
                    'sb': 0,
                    'lb': 1,
                    'bl': 2,
                    'sl': 3,
                    'fl': 4,
                    'lf': 5,
                    'sf': 6,
                    'rf': 7,
                    'fr': 8,
                    'sr': 9,
                    'br': 10,
                    'rb': 11
                }

            # used for deriving NLP annotations
            tpcc_dict_inv = {v: k for k, v in tpcc_dict.items()}

            import math        
            PI = math.pi

            for i in range(0, height):
                for j in range(0, width):
                    d_x = j - x_odom
                    d_y = -1 * (i - y_odom)

                    #print('\n(i, j) = ', (i, j))
                    
                    #k = (-plan_y_list[-1] + y_odom) / (d_x)
                    k = d_y / max(1, d_x)
                    #print('abs(k) = ', abs(k))
                    # angle in radians
                    angle = np.arctan2(d_y, np.sign(d_x) * max(1, abs(d_x)))
                    #print('angle (in degrees) = ', angle * 180 / math.pi)
                    
                    r_ = (i - relatum[1])**2 + (j - relatum[0])**2
                    r_ = math.sqrt(r_)
                    #print('(r_, r) = ', (r_, r))

                    if r > 0:
                        # starting with 'c'
                        if r_ <= r:
                            value = 'c'
                        # starting with 'd'
                        else:
                            value = 'd'
                    elif r == 0:
                        value = ''

                    #print('angle = ', angle)
                    if angle == 0:
                        value += 'sr'
                    elif 0 < angle <= PI/4:
                        value += 'fr'
                    elif PI/4 < angle < PI/2:
                        value += 'rf'
                    elif angle == PI/2:
                        value += 'sf'
                    elif PI/2 < angle < 3*PI/4:
                        value += 'lf'
                    elif 3*PI/4 <= angle < PI:
                        value += 'fl'
                    elif angle == PI:
                        value += 'sl'
                    elif -PI < angle <= -3*PI/4:
                        value += 'bl'
                    elif -3*PI/4 < angle < -PI/2:
                        value += 'lb'
                    elif angle == -PI/2:
                        value += 'sb'
                    elif -PI/2 < angle < -PI/4:
                        value += 'rb'        
                    elif -PI/4 <= angle < 0:
                        value += 'br'                            

                    segments[i, j] = tpcc_dict[value]    

            # reasoning for one random position
            referent = [random.randint(0, width-1), random.randint(0, height-1)]
            print('\nreferent = ', referent)
            print('segments[referent] = ', segments[referent[1],referent[0]])
            if referent == relatum == origin:
                print('origin, relatum (tri, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif origin == relatum:
                print('origin, relatum (dou, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif referent == relatum:
                print('origin, relatum (sam, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            else:
                print('origin, relatum ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ' referent')

        elif semantic_choice == 3:
            # A model [(Herrmann, 1990),(Hernandez, 1994)] from 'moratz2002spatial'
            # used for getting semantic costmap
            tpcc_dict = {
                'front': 0,
                'left': 1,
                'back': 2,
                'right': 3,
                'left-front': 4,
                'left-back': 5,
                'right-front': 6,
                'right-back': 7,
                'straight-front': 8,
                'exactly-left': 9,
                'straight-back': 10,
                'exactly-right': 11
            }

            # used for deriving NLP annotations
            tpcc_dict_inv = {v: k for k, v in tpcc_dict.items()}

            import math        
            PI = math.pi

            for i in range(0, height):
                for j in range(0, width):
                    d_x = j - x_odom
                    d_y = -1 * (i - y_odom)

                    #print('\n(i, j) = ', (i, j))
                    
                    #k = (-plan_y_list[-1] + y_odom) / (d_x)
                    k = d_y / max(1, d_x)
                    #print('abs(k) = ', abs(k))
                    # angle in radians
                    angle = np.arctan2(d_y, np.sign(d_x) * max(1, abs(d_x)))
                    #print('angle (in degrees) = ', angle * 180 / math.pi)

                    value = ''

                    #print('angle = ', angle)
                    if -PI/4 <= angle <= PI/4:
                        value += 'right'
                    elif PI/4 < angle < 3*PI/4:
                        value += 'front'
                    elif 3*PI/4 <= angle or angle <= -3*PI/4:
                        value += 'left'
                    elif -3*PI/4 < angle < -PI/4:
                        value += 'back'                            

                    segments[i, j] = tpcc_dict[value]    

            # reasoning for one random position
            referent = [random.randint(0, width-1), random.randint(0, height-1)]
            print('\nreferent = ', referent)
            print('segments[referent] = ', segments[referent[1],referent[0]])
            if referent == relatum == origin:
                print('origin, relatum (tri, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif origin == relatum:
                print('origin, relatum (dou, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif referent == relatum:
                print('origin, relatum (sam, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            else:
                print('origin, relatum ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ' referent')

        elif semantic_choice == 4:
            # Model for combined expressions from 'moratz2002spatial'
            # used for getting semantic costmap
            tpcc_dict = {
                'front': 0,
                'left': 1,
                'back': 2,
                'right': 3,
                'left-front': 4,
                'left-back': 5,
                'right-front': 6,
                'right-back': 7,
                'straight-front': 8,
                'exactly-left': 9,
                'straight-back': 10,
                'exactly-right': 11
            }

            # used for deriving NLP annotations
            tpcc_dict_inv = {v: k for k, v in tpcc_dict.items()}

            import math        
            PI = math.pi

            for i in range(0, height):
                for j in range(0, width):
                    d_x = j - x_odom
                    d_y = -1 * (i - y_odom)

                    #print('\n(i, j) = ', (i, j))
                    
                    #k = (-plan_y_list[-1] + y_odom) / (d_x)
                    k = d_y / max(1, d_x)
                    #print('abs(k) = ', abs(k))
                    # angle in radians
                    angle = np.arctan2(d_y, np.sign(d_x) * max(1, abs(d_x)))
                    #print('angle (in degrees) = ', angle * 180 / math.pi)

                    value = ''

                    #print('angle = ', angle)
                    if angle == 0:
                        value += 'exactly-right'
                    elif 0 < angle < PI/2:
                        value += 'right-front'
                    elif angle == PI/2:
                        value += 'straight-front'
                    elif PI/2 < angle < PI:
                        value += 'left-front'
                    elif angle == PI:
                        value += 'exactly-left'
                    elif -PI < angle < -PI/2:
                        value += 'left-back'
                    elif angle == -PI/2:
                        value += 'straight-back'
                    elif -PI/2 < angle < 0:
                        value += 'right-back'                            

                    segments[i, j] = tpcc_dict[value]    

            # reasoning for one random position
            referent = [random.randint(0, width-1), random.randint(0, height-1)]
            print('\nreferent = ', referent)
            print('segments[referent] = ', segments[referent[1],referent[0]])
            if referent == relatum == origin:
                print('origin, relatum (tri, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif origin == relatum:
                print('origin, relatum (dou, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            elif referent == relatum:
                print('origin, relatum (sam, ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ') referent')
            else:
                print('origin, relatum ' + tpcc_dict_inv[segments[referent[1],referent[0]]] + ' referent')
            
        segmentation_plot = True
        if segmentation_plot == True:
            dirName = 'explanation_results'
            try:
                os.mkdir(dirName)
            except FileExistsError:
                pass
            
            big_plot_size = True
            w = h = 1.6
            if big_plot_size == True:
                w *= 3
                h *= 3

            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(segments.astype('float64'), aspect='auto')
            fig.savefig(self.dirCurr + '/' + dirName + '/segments_final.png', transparent=False)
            fig.clf()

        #return np.zeros(image.shape, np.uint8)
        return segments


    # EVALUATION    
    def explain_instance_evaluation(self, image, classifier_fn, costmap_info, map_info, tf_odom_map, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_segments=1,
                         num_samples_current=1,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):

        import copy
        image_orig = copy.deepcopy(image)                 

        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        # my change
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
            segments = segmentation_fn(image)
        elif segmentation_fn == 'custom_segmentation':
            start = time.time()
            segments = self.sm_slic_custom(image_orig, image)
            #segments = self.sm_only_obstacles(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
            #segments = self.sm_only_obstacles_new(image_orig, image, x_odom, y_odom, devDistance_x, sum_x, devDistance_y, sum_y, devDistance, plan_x_list, plan_y_list)
            #segments = self.sm_semantic(image, x_odom, y_odom)
            end = time.time()
            segmentation_time = end - start
            #segmentation_time = round(end - start, 3)
        else:
            segments = segmentation_fn(image)

        fudged_image = image_orig.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels, classifier_fn_time, planner_time, target_calculation_time, costmap_save_time = self.data_labels_evaluation(image_orig, fudged_image, segments,
                                        classifier_fn, num_segments, num_samples_current,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances_start = time.time()
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()
        distances_end = time.time()
        distances_time = distances_end - distances_start

        regressor_start = time.time()
        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        regressor_end = time.time()
        regressor_time = regressor_end - regressor_start

        return ret_exp, segments, segmentation_time, classifier_fn_time, planner_time, target_calculation_time, costmap_save_time, distances_time, regressor_time

    def data_labels_evaluation(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_segments,
                    num_samples_current,
                    batch_size=10,
                    progress_bar=True):

        num_samples = 2 ** (num_segments+1)
        #print('num_samples: ', num_samples)
        import itertools
        lst = list(map(list, itertools.product([0, 1], repeat=(num_segments+1))))
        data = np.array(lst).reshape((num_samples, num_segments+1))
        #print('data.shape[0] - original - original: ', data.shape[0])
        
        labels = []

        show_free_space = False

        if show_free_space == True:
            data[0, :] = 1
            data[-1, :] = 0 # only if I use my perturbation
        else:
            data = data[int(data.shape[0]/2):, :]
            data[0, 1:] = 1
            data[-1, 1:] = 0    

        labels = []

        #print('data.shape[0] - original: ', data.shape[0])
        if len(data) != num_samples_current:
            data_copy = copy.deepcopy(data)
            data = data_copy[np.random.choice(len(data_copy), num_samples_current, replace=False)]
            data[0, :] = 1
        #print('num_segments: ', num_segments)
        #print('num_segments_current: ', num_segments_current)
        #print('data.shape[0]: ', data.shape[0])    

        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                #start = time.time()
                preds = classifier_fn(np.array(imgs))
                #end = time.time()
                #classifier_fn_time = round(end - start, 3)
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            start = time.time()
            preds, planner_time, target_calculation_time, costmap_save_time = classifier_fn(np.array(imgs))
            end = time.time()
            classifier_fn_time = end - start
            #classifier_fn_time = round(end - start, 3)
            labels.extend(preds)

        #print('data_labels ends')

        return data, np.array(labels), classifier_fn_time, planner_time, target_calculation_time, costmap_save_time
