#!/usr/bin/env python3
# coding: utf-8

"""
Viola Jones classifier.
Copyright (c) 2017 Mikhail Okunev (mishka.okunev@gmail.com)
Copyright (c) 2017 Paul Beltyukov (beltyukov.p.a@gmail.com)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#==============================================================================
import pickle

import numpy as np

import os
import os.path
from os import walk

import progressbar

from skimage import io

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors as cl

import sys
sys.path.append(os.getcwd())
from libvj import *

import random
from random import randint



#==============================================================================
def _get_all_images(starting_dir):
    images = []
    extensions = ["pgm", "jpeg", "jpg", "png"]
    for dir, _, filenames in walk(starting_dir):
        for filename in filenames:
            extension = os.path.splitext(filename)[1][1:]
            if extension in extensions:
                image = io.imread(os.path.join(dir, filename))
                images.append(image)
    return images

#==============================================================================
def get_all_images(starting_dir, img_file):
    
    if os.path.isfile(img_file):
        return np.load(img_file)
    else:
        ret = _get_all_images(starting_dir)
        np.save(img_file, ret)
        return ret

#==============================================================================
# Препроцессинг изображений с лицами
# 
# * Нормируем яркость, чтобы не учитывать освещенность
# * Преобразуем к 24 * 24

def prepare_positives(images, result_l):
    norm_images = [normalize_image(im.astype('float')) for im in images]
    resized_images = [resize(im, (result_l, result_l), mode='constant') for im in norm_images]
    return resized_images


#==============================================================================
# Препроцессинг изображений без лиц
# 
# * Вырежем случайные квадраты из негативных изображений
# * Нормируем яркость
# * Преобразуем к 24 * 24

def prepare_negatives(images, sample_size, result_l):
    norm_images = [normalize_image(im.astype('float')) for im in images]
    crops = []
    for _ in range(0, sample_size):
        image_ind = randint(0, len(norm_images) - 1)
        image = norm_images[image_ind]
        w, h = image.shape
        max_r = min(w, h)
        r = random.randint(result_l, max_r)
        x, y = randint(0, w - max_r), randint(0, h - max_r)
        crop = image[x: x + r, y: y + r]
        crop = resize(crop, (result_l, result_l), mode='constant')
        crops.append(crop)
    return crops

#==============================================================================
print('Loading images...')
positives = get_all_images('data/positives', 'data/pos_img.npy')
n_positives = len(positives)
negatives = get_all_images('data/negatives', 'data/neg_img.npy')
print('Done!')

#==============================================================================
# Зафиксируем размер окна, на котором будет работать классификатор
image_canonical_size = 24


#==============================================================================
pos_prep_fl = 'data/pos_prep.npy'

print('Prepare positive images...')
if os.path.isfile(pos_prep_fl):
    positives_prepared = np.load(pos_prep_fl)
else:
    positives_prepared = prepare_positives(positives, image_canonical_size)
    np.save(pos_prep_fl, positives_prepared)
print('Done!')


#==============================================================================
# Возьмем столько же негативных изображений, сколько позитивных
n_negatives = n_positives

#==============================================================================
neg_prep_fl = 'data/neg_prep.npy'

print('Preparing negative images...')
if os.path.isfile(neg_prep_fl):
    negatives_prepared = np.load(neg_prep_fl)
else:
    negatives_prepared = prepare_negatives(negatives, n_negatives, image_canonical_size)
    np.save(neg_prep_fl, negatives_prepared)
print('Done!')

#==============================================================================
# Проверим, что данные имеют правильный формат
def image_has_correct_format(image, shape=(image_canonical_size, image_canonical_size)):
    return image.shape == shape

assert(len(positives_prepared) == n_positives)
assert(all([image_has_correct_format(im) for im in positives_prepared]))

assert(len(negatives_prepared) == n_negatives)
assert(all([image_has_correct_format(im) for im in negatives_prepared]))


#==============================================================================
print('Preparing integral images...')
integral_positives = get_integral_imgs(positives_prepared, 'data/pos_int.npy') #[IntegralImage(im) for im in positives_prepared]
integral_negatives = get_integral_imgs(negatives_prepared, 'data/neg_int.npy') #[IntegralImage(im) for im in negatives_prepared]
print('Done!')


#==============================================================================
# Сохраним все возможные признаки

features_to_use = [HaarFeatureVerticalTwoSegments, 
                   HaarFeatureVerticalThreeSegments, 
                   HaarFeatureHorizontalTwoSegments,
                   HaarFeatureHorizontalThreeSegments,
                   HaarFeatureFourSegments]
# шаги по x,y,w,h
x_stride = 2
y_stride = 2
w_stride = 2
h_stride = 2

all_features = []
for x in range(0, image_canonical_size, x_stride):
    for y in range(0, image_canonical_size, y_stride):
        for w in range(2, image_canonical_size - x + 1, w_stride):
            for h in range(2, image_canonical_size - y + 1, h_stride):
                for feature_type in features_to_use:
                    try:
                        feature = feature_type(x, y, w, h)
                        all_features.append(feature)
                    except:
                        continue
print("Всего признаков: {}".format(len(all_features)))  

#==============================================================================
def _compute_features(integral_images, features):
    result = np.zeros((len(integral_images), len(features)))
    bar = progressbar.ProgressBar(maxval = len(integral_images))
    
    bar.start()
    for ind, integral_image in enumerate(bar(integral_images)):
        result[ind] = compute_features_for_image(integral_image, features)
        bar.update(ind+1)
    return result

#==============================================================================
def compute_features(integral_images, features, ftr_file):
    if os.path.isfile(ftr_file):    
        return np.load(ftr_file)
    else:
        ret = _compute_features(integral_images, features)
        np.save(ftr_file, ret)
        return ret

#==============================================================================
print('Will compute features...')
positive_features = compute_features(integral_positives, all_features, 'data/pos.npy')
negative_features = compute_features(integral_negatives, all_features, 'data/neg.npy')
print('Done!')

#==============================================================================
# Подготовим тренировочный набор

print('Will prepare train set...')
y_positive = np.ones(len(positive_features))
y_negative = np.zeros(len(negative_features))
    
X_train = np.concatenate((positive_features, negative_features))
y_train = np.concatenate((y_positive, y_negative))
print('Done!')

#==============================================================================

print('Will get face detector!')

fd_file = 'data/face_detector.pickle'

if os.path.isfile(fd_file):
    
    print('Loading...')        
    vj_cls = pickle.load(open(fd_file,'rb'))
else:
    
    vj_cls = ViolaJonesСlassifier(rounds = 100)
    
    print('Will train face detector...')
    vj_cls.fit(X_train, y_train)
    print('Will optimize face detector...')
    vj_cls.add_features(all_features)

    print('Will check for false positive...')
    negatives_prepared_new = prepare_negatives(negatives, 10000, image_canonical_size)
    pred_neg_new = vj_cls.classify_wlist(negatives_prepared_new)
    false_positive_rate = sum(pred_neg_new) / len(pred_neg_new)
    print("Процент ложных обнаружений: {}".format(false_positive_rate * 100))

    print('Will check for positive...')
    test = get_all_images('data/test', 'test_img.npy')
    test_prepared = prepare_positives(test, image_canonical_size)
    test_positives_result = vj_cls.classify_wlist(test_prepared)
    detection_rate = sum(test_positives_result) / len(test_positives_result)
    print("Процент корректных обнаружений: {}".format(detection_rate * 100))
    
    print('Will calibrate face detector...')
    vj_cls.calibrate(test_prepared, negatives_prepared_new)    
    
    print('Will save the face detector...')
    pickle.dump(vj_cls, open(fd_file,'wb'))
    
print('Done!')

#==============================================================================
# Воспользуемся полученным классификатором, чтобы найти лица на изображении
images_to_scan = get_all_images('data/for_scanning', 'data/for_scan_img.npy')
#==============================================================================
result = vj_cls.detect(images_to_scan[0], image_canonical_size)

np.save('detected_frames.npy', np.array(result))

im = images_to_scan[0]

fig,ax = plt.subplots(1)
fig.set_size_inches(20,20)
ax.imshow(im, cmap='gray')

for x, y, xc, yc, qa in result:
    ecl = cl.hsv_to_rgb(np.array([(qa-1)*2, 1.0, 1.0]))
    rect = patches.Rectangle((y,x),yc - y,xc - x,linewidth=1,edgecolor=ecl,facecolor='none')
    ax.add_patch(rect)

plt.show()

