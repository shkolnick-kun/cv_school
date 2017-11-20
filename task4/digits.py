#!/usr/bin/env python
# -*- coding: utf-8 -*-s

"""
MNIST digits classifier.
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

from skimage.feature import hog

from sklearn.datasets import fetch_mldata
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
#==============================================================================
# В этом классе используется код из VotingClassifier, поэтому чисто для него:
# Authors: Sebastian Raschka <se.raschka@gmail.com>,
#          Gilles Louppe <g.louppe@gmail.com>,
#          Paul Beltyukov <beltyukov.p.a@gmail.com>
#
# Licence: BSD 3 clause
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class SimpleVotingClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_

        return self

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators]).T

    def predict(self, X):

        predictions = self._predict(X).astype('int64')

        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)

        maj = self.le_.inverse_transform(maj)
        
        return maj

#==============================================================================
cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
#==============================================================================
# Делаем HOG для квадрата 2*2 или 4*4 и 7*7, стобы не расширять до 32*32
# Оптимальный размер 4*4, дает сразу > 95%, 2*2 - мало, 7*7 съедает полезную информацию
hog_sz = 4
#==============================================================================
def mnist_hog(img):
    return hog(img, orientations=8, pixels_per_cell=(hog_sz, hog_sz), cells_per_block=(1, 1))

#==============================================================================
# Делаем HOG для списка рисунков и сохраняем в кеш
def hog_do_and_save(img, hfile):

    # Начало работы
    print "Do HOG..."

    # Вычисляем
    for i in range(0, len(img)):
        img[i] = mnist_hog(img[i])
        print i

    # Соханяем в кеш
    print "Save to file"

    np.save(hfile, img)

    return img

#==============================================================================
# Поворачиваем рисунки
def img_rot_and_save(images, labels, ifile):
    print 'Rotate images'
    img = []
    lbl = []
    for angle in range(-45, 60, 45):

        print angle

        # Количество меток увеличилось!
        lbl = lbl + list(labels)

        # Вращаем вокруг центра изображения
        M = cv2.getRotationMatrix2D((14, 14), float(angle), 1.0)

        for i in range(0, len(images)):
            img.append(cv2.warpAffine(images[i], M, images[i].shape))

    # Сохраняем в кеш
    np.save(ifile, (img, lbl))

    return img, lbl
#==============================================================================
# На самом деле точность и так была 95,75%, безо всякого вращения
img_file = os.path.join(cache_dir, 'img_data.npy')
if os.path.isfile(img_file):
    # Грузим рисунки из кеша
    print 'Load images and labels...'
    img, lbl = np.load(img_file)

else:
    # В кеше нет изображений, считаемм по новой
    mnist     = fetch_mldata('MNIST original', data_home = cache_dir)    
    
    # Делаем из них рисунки для HOG
    images = list(np.reshape(mnist.data, (len(mnist.data), 28, 28)))

    # Вращаем и сохраняем в кеше
    img, lbl = img_rot_and_save(images, mnist.target, img_file)

images = img
lbl = np.array(lbl, 'int64')
print 'Done!'

# Получаем изображения, оработанные HOG
hog_file = os.path.join(cache_dir, 'hog_data.npy')
if os.path.isfile(hog_file):

    # Грузим из кеша
    print 'Load HOG...'
    cached_img = np.load(hog_file)

    # Проверка размера HOG - дескрипторов
    if len(cached_img[0]) == 8 * (28 / hog_sz)*(28 / hog_sz):

        # Размер соответствует текущему размеру ячейки
        img = cached_img
    else:

        # Размер не соответствует текущему размеру ячейки, надо пересчитывать!
        print 'HOG data outdated!'
        img = hog_do_and_save(img, hog_file)
else:
    img = hog_do_and_save(img, hog_file)

# Теперь у нас есть HOG
print "Done!"

#images_sample = resample(images, replace = False, n_samples = 40)

X_train, X_test, y_train, y_test = train_test_split(img, lbl, test_size = 1.0 / 5, random_state = 42)

# Ищем оптимальные параметры обучения с помощью кросс-валидации
kf = StratifiedKFold(y_train, n_folds = 5, shuffle = True)

# Список параметров


cval_par =  [
                ('l1', 0.1), ('l1', 0.5), ('l1', 1.0), ('l1', 5.0), ('l1', 10.0), ('l1', 50.0), ('l1', 100.0), 
                ('l2', 0.1), ('l2', 0.5), ('l2', 1.0), ('l2', 5.0), ('l2', 10.0), ('l1', 50.0), ('l2', 100.0) 
            ]

# TODO: Закомментарить перед отправкой на проверку

cval_par =  [
                ('l2', 50.0)
            ]
#"""

# Список точностей
acc = []

# Цикл по параметрам
for pnl, rgw in cval_par:
    
    print 'Learning params:\'', pnl, '\':', rgw
    acc_avg = []

    # Разбили выборку разыми способами
    for i_trn, i_tst in kf:

        # Обучаем модель
        print 'TRAIN:', len(i_trn), 'TEST:', len(i_tst)
        model = LogisticRegression(penalty = pnl, tol = 0.00001, C = rgw, solver = 'liblinear', max_iter = 5)
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
pnl, rgw = cval_par[i_opt]
print pnl, rgw

# Финальное обучение модели
# Дает 95,75% при HOG 4*4
model = LogisticRegression(penalty = pnl, tol = 0.00001, C = rgw, solver = 'liblinear', max_iter = 10)
# Дает 95,3%
#model = LogisticRegression(penalty='l2', tol=0.000001, C=1000.0, solver='lbfgs', max_iter=20, multi_class='multinomial')
"""
# Дает 95,8%, пока нет смысла заморачиваться
# Анонимные эксперты проявляют конформизм, не поучается существенно повысить достоверность результатов

# А теперь попробуем обучить коммитет анонимных экспертов 
kf = StratifiedKFold(y_train, n_folds = 5, shuffle = True)
experts = []
#expw = []
for i_trn, i_tst in kf:

    # Обучаем модель
    print 'TRAIN:', len(i_trn), 'TEST:', len(i_tst)
    model = LogisticRegression(penalty = pnl, tol = 0.00001, C = rgw, solver = 'liblinear', max_iter = 10)
    model.fit(X_train[i_trn], y_train[i_trn])

    # Считаем точность
    print model.score(X_train[i_tst], y_train[i_tst])

    experts.append(model)
    
    #experts.append(('exp'+str(len(experts)), model))
    #expw.append(1)
        
print experts

# Нет, я не читер, я просто хороший маляр!
model = SimpleVotingClassifier(experts)
# """
print 'Will train the final model...'
model.fit(X_train, y_train)
print 'Done!'

print 'Model accuracy:', model.score(X_test, y_test)

# Ищем рисунки, на которых модель выдает ошибки
print 'Will save error images...'
y_predict = model.predict(img)

err_dir = os.path.join(cache_dir, 'err')
if not os.path.isdir(err_dir):
    os.mkdir(err_dir)

for i in range(0, len(y_predict)):
    # Нас интересуют ошибки
    if y_predict[i] != lbl[i]:
        # Запомнинаем исходный образ
        cv2.imwrite(os.path.join(err_dir, 'img_' + str(int(lbl[i])) + '_' + str(i) + '.png'), images[i])

print 'Done!'

# Применяем KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)

print 'Will train KNeighbors...'
neigh.fit(X_train, y_train)
print 'Done!'

# Очень медленно считает!!!
print 'Will classify with KNeighbors...'
print 'KNeighbors accuracy:', neigh.score(X_test, y_test)
print 'Done!'

#TODO: Аугментировать тренировочный набор искуственными изображениями
#TODO: Запустить свою модель на собственном почерке
