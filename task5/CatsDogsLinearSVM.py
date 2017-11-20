#!/usr/bin/env python
# -*- coding: utf-8 -*-s

"""
Cat vs Dog classifier.
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

import sys
import os
import cv2
import numpy as np

from skimage import io, color, transform, exposure
from skimage.feature import hog

from sklearn.utils import resample

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC

# OS: Linux Mint 18.2
# Python  2.7.x
# OpenCV  2.4.x
# sklearn 0.17.x
# skimage 0.10.x
#==============================================================================
def make_dir(target):
    if not os.path.isdir(target):
        os.mkdir(target)    

#==============================================================================
hog_sz = 8
#==============================================================================
def get_hog(img):
    return hog(img, orientations=8, pixels_per_cell=(hog_sz, hog_sz), cells_per_block=(2, 2), normalise=True)

#==============================================================================
# Тут будут лежать разного рода полезные данные, например, картинки, на которых мы ошибаемся
cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
# Датасет тоже положу туда
train_dir = os.path.join(cache_dir,'train')
#==============================================================================
# Делаем HOG для списка рисунков и сохраняем в кеш
def dataset_prepare(hfile):

    # Начало работы
    print "Preparing dataset..."

    i = 0
    img = []
    hog = []
    lbl = []
    for fname in os.listdir(train_dir):

        # Проверка входных файлов
        if not(fname.endswith(".png") or fname.endswith(".bmp") or fname.endswith(".jpg")):
            continue

        # Читаем файлы
        fl = os.path.join(train_dir, fname)        
        src = io.imread(fl)

        # Ресайз
        src = transform.resize(src, (64, 64))

        # Как насчет Гаусса здесь?
        #src = gaussian_filter(src, sigma = 1.0)

        # Конвертируем в LAB - лучше воспроизводит человеческое восприятие
        src = color.rgb2lab(src)
        l = src[:,:,0]
        a = src[:,:,1]
        b = src[:,:,2]

        # Нормализация
        l = exposure.rescale_intensity(l)

        # Преобразуем в массивы
        l = np.asarray(l, 'float64')
        a = np.asarray(a, 'float64')
        b = np.asarray(b, 'float64')

        # Получаем HOG
        l_ftr = get_hog(l)

        # Получаем цветовые особенности
        H,W   = l.shape
        a_ftr = a[hog_sz/2: H: hog_sz, hog_sz/2: W: hog_sz].copy().flatten()
        b_ftr = b[hog_sz/2: H: hog_sz, hog_sz/2: W: hog_sz].copy().flatten()

        # Получаем вектор фич
        src_ftr = np.array(list(l_ftr) + list(a_ftr) + list(b_ftr), 'float64')

        # Коррекция на всякий случай
        s = np.sum(src_ftr)
        if np.isnan(s) or np.isinf(s) or np.isneginf(s):
            print 'Sample is grayscaled:', i, 'Will fix feature vector!'
            for j in range(0, len(src_ftr)):
                s = src_ftr[j] 
                if np.isnan(s) or np.isinf(s) or np.isneginf(s):
                    src_ftr[j] = 0.0
        
        # Добавляем рисунок
        img.append(l)

        # Добавляем Фичи
        hog.append(src_ftr)

        # Добавляем метки
        if 'dog' in fname:
            lbl.append(0)
        else:
            lbl.append(1)

        # Индикация работы программы
        if i % 100 == 0:
            print 'File number:', i, 'File name:', fname
        i = i + 1
    
    # Формируем датасет
    dataset = (np.array(img), np.array(hog), np.array(lbl))     

    # Соханяем в кеш
    print "Save to file..."
    np.savez(hfile, img = dataset[0], hog = dataset[1], lbl = dataset[2])

    return dataset

#==============================================================================
def get_dataset(hfile):
    if os.path.isfile(hfile):
        # Грузим датасет из кеша
        print 'Loading dataset...'

        # т.к. сохраняли с помощью np.savez, дотавать надо будет по именам
        data = np.load(hfile)
        dataset = (data['img'], data['hog'], data['lbl'])
        data.close()

        return dataset

    else:
        # Готовим датасет
        return dataset_prepare(hfile)


#==============================================================================
# Получение датасета для последующего обучения классификатора
print 'Will get dataset...'
img, hog, lbl = get_dataset(os.path.join(cache_dir, 'dataset_'+str(hog_sz)+'.npz'))
print 'Done!'

print 'Scaling fetures...' 
scaler = MinMaxScaler()
#scaler = StandardScaler()
hog = scaler.fit_transform(hog)
print 'Done!'


print 'Will brake dataset to train/test parts...'
X_train, X_test, y_train, y_test = train_test_split(hog, lbl, test_size = 1.0 / 5, random_state = 42)

print 'Done!'

print 'Will find regularization parameter...'
# Дает около 70%, это близко к потолку для линейных SVM, т.к. мы используем тупо HOG, даже не Dense-SIFT 
#lin_mdl = LinearSVC(C = 0.7, loss = 'hinge', max_iter = 2000, tol = 0.001)

# Ищем оптимальные параметры обучения с помощью кросс-валидации
kf = StratifiedKFold(y_train, n_folds = 5, shuffle = True)

# Параметр C
#c_var = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
c_var = (0.05, 0.1, 0.2)#0.5, 1.0, 2.0, 5.0, 10.0)
# Список точностей
acc = []

# Перебор параметра
for c_cur in c_var:
    
    print 'Learning params: C = ', c_cur
    acc_avg = []

    # Разбили выборку разыми способами
    for i_trn, i_tst in kf:

        # Обучаем модель
        print 'TRAIN:', len(i_trn), 'TEST:', len(i_tst)

        model = LinearSVC(C = c_cur, max_iter = 500, tol = 0.001)
        model.fit(X_train[i_trn], y_train[i_trn])

        # Считаем точность
        val_acc = model.score(X_train[i_tst], y_train[i_tst])
        print val_acc
        acc_avg.append(val_acc)

    # Считаем среднюю точность для данного набора параметров 
    avg = np.average(np.array(acc_avg))
    print avg
    acc.append(avg)

# Находим оптимальный набор параметров
i_opt = np.argmax(acc)
c_opt = c_var[i_opt]
print 'C =', c_opt
print 'Done!'

print 'Will train linear model...'
lin_mdl = LinearSVC(C = c_opt, max_iter = 2000, tol = 0.0005)
lin_mdl.fit(X_train, y_train)
print 'Linear model accuracy:', lin_mdl.score(X_test, y_test)
print 'Done!'
