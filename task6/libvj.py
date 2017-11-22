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
import abc
import math

import numpy as np

import os
import os.path

import progressbar

from skimage.transform import resize

#==============================================================================
# Вычтем из изображения среднее и поделим на стандартное отклонение
def normalize_image(image):
    mean, std = image.mean(), image.std()
    return ((image - mean) / std)

#==============================================================================
# Интегральное изображение

class IntegralImage:
    def __init__(self, image):
        # hint: воспользуйтесь numpy.cumsum два раза, чтобы получить двумерную кумулятивную сумму
        h,w = image.shape
        
        ii = np.zeros((h + 1, w + 1), image.dtype)
        
        ii[1:, 1:] = np.cumsum(np.cumsum(image,0),1)
        
        self.integral_image = ii
    
    def sum(self, x1, y1, x2, y2):
        '''
        Сумма подмассива
        
        На входе:
            x1, y1 -- координаты левого нижнего угла прямоугольника запроса
            x2, y2 -- координаты верхнего правого угла прямоугольника запроса
            
        На выходе:
            Сумма подмассива [x1..x2, y1..y2]
        '''
        assert(x1 <= x2)
        assert(y1 <= y2)
        
        x2 = x2 + 1
        y2 = y2 + 1
        
        b11 = self.integral_image[x1, y1]
        b12 = self.integral_image[x2, y1]
        b21 = self.integral_image[x1, y2]
        b22 = self.integral_image[x2, y2]
        
        return b22 - b12 - b21 + b11

#==============================================================================
def get_integral_imgs(imgs, img_file):
    
    if os.path.isfile(img_file):
        return list(np.load(img_file))
    else:
        ret = [IntegralImage(im) for im in imgs]
        np.save(img_file, np.array(ret))
        return ret

#==============================================================================
# Признаки Хаара
#------------------------------------------------------------------------------
# Общий интерфейс для всех классов признаков

class HaarFeature(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def compute_value(self, integral_image):
        '''
        Функция, вычисляющая и возвращающая значение признака
        
        На входе:
            integral_image -- IntegralImage
            
        На выходе:
            Значение признака
        '''
        pass
    
    def __repr__(self):
        return "Feature {}, {}, {}, {}".format(self.x_s, self.y_s, self.x_e, self.y_e)


#==============================================================================
class HaarFeatureVerticalTwoSegments(HaarFeature):
    def __init__(self, x, y, w, h):
        assert(h % 2 == 0)
        assert(x >= 0)
        assert(y >= 0)
        assert(w >= 2)
        assert(h >= 2)
        
        self.x_s = x
        self.x_e = x + w - 1
        
        self.y_s   = y
        self.y_m   = y + h // 2
        self.y_m_1 = y + h // 2 - 1
        self.y_e   = y + h - 1
        
    def compute_value(self, integral_image):
        s1 = integral_image.sum(self.x_s, self.y_s, self.x_e, self.y_m_1)
        s2 = integral_image.sum(self.x_s, self.y_m, self.x_e, self.y_e)
        return s1 - s2

#==============================================================================
class HaarFeatureVerticalThreeSegments(HaarFeature):
    
    def __init__(self, x, y, w, h):
        assert(h % 3 == 0)
        assert(x >= 0)
        assert(y >= 0)
        assert(w >= 2)
        assert(h >= 3)
        
        self.x_s  = x
        self.x_e  = x + w - 1
    
        self.y_s  = y
        self.y_m1 = y + h // 3
        self.y_m2 = y + 2 * h // 3 - 1
        self.y_e  = y + h - 1
        
    def compute_value(self, integral_image):
        s1 = integral_image.sum(self.x_s, self.y_s,  self.x_e, self.y_e)
        s2 = integral_image.sum(self.x_s, self.y_m1, self.x_e, self.y_m2)
        return s1 - 2.0 * s2

#==============================================================================
class HaarFeatureHorizontalTwoSegments(HaarFeature):
    
    def __init__(self, x, y, w, h):
        assert(h % 2 == 0)
        assert(x >= 0)
        assert(y >= 0)
        assert(w >= 2)
        assert(h >= 2)
        
        self.x_s   = x
        self.x_m   = x + w // 2
        self.x_m_1 = x + w // 2 - 1
        self.x_e   = x + w - 1
        
        self.y_s   = y
        self.y_e   = y + h - 1
        
    def compute_value(self, integral_image):
        s1 = integral_image.sum(self.x_m, self.y_s, self.x_e,   self.y_e)
        s2 = integral_image.sum(self.x_s, self.y_s, self.x_m_1, self.y_e)
        return s1 - s2

#==============================================================================
class HaarFeatureHorizontalThreeSegments(HaarFeature):
        
    def __init__(self, x, y, w, h):
        assert(w % 3 == 0)
        assert(x >= 0)
        assert(y >= 0)
        assert(h >= 2)
        assert(w >= 3)
        
        self.y_s  = y
        self.y_e  = y + h - 1
    
        self.x_s  = x
        self.x_m1 = x + w // 3
        self.x_m2 = x + 2 * w // 3 - 1
        self.x_e  = x + w - 1
        
    def compute_value(self, integral_image):
        s1 = integral_image.sum(self.x_s,  self.y_s,  self.x_e,  self.y_e)
        s2 = integral_image.sum(self.x_m1, self.y_s,  self.x_m2, self.y_e)
        return s1 - 2*s2

#==============================================================================
class HaarFeatureFourSegments(HaarFeature):
    def __init__(self, x, y, w, h):
        assert(h % 2 == 0)
        assert(w % 2 == 0)
        assert(x >= 0)
        assert(y >= 0)
        assert(w >= 2)
        assert(h >= 2)
        
        self.x_s   = x
        self.x_m   = x + w // 2
        self.x_m_1 = x + w // 2 - 1
        self.x_e   = x + w - 1
        
        self.y_s   = y
        self.y_m   = y + h // 2
        self.y_m_1 = y + h // 2 - 1
        self.y_e   = y + h - 1
        
    def compute_value(self, integral_image):
        
        s1 = integral_image.sum(self.x_s, self.y_s, self.x_e,   self.y_e  )
        s2 = integral_image.sum(self.x_s, self.y_m, self.x_m_1, self.y_e  )
        s3 = integral_image.sum(self.x_m, self.y_s, self.x_e,   self.y_m_1)
        
        return s1 - 2*s2 - 2*s3
       
#==============================================================================
# Вычислим все признаки на всех изображениях

def compute_features_for_image(integral_image, features):
    result = np.zeros(len(features))
    for ind, feature in enumerate(features):
        result[ind] = feature.compute_value(integral_image)
    return result

#==============================================================================
# Базовый классификатор

class DecisionStump:
    def __init__(self, threshold = 0, polarity = 1):
        self.threshold = threshold
        self.polarity = polarity
        
    def train(self, X, y, w, indices):
        '''
            Функция осуществляет обучение слабого классификатора
            
            На входе:
                X -- одномерный отсортированный numpy массив со значениями признака
                y -- одномерный numpy массив со значением класса для примера (0|1)
                Порядок y -- до сортировки X
                w -- одномерный numpy массив со значением весов признаков
                Порядок w -- до сортировки X
                indices -- одномерный numpy массив, перестановка [несортированный X] -> [сортированный X]
                Массив indices нужен для оптимизации,
                чтобы не сортировать X каждый раз, мы предсортируем значения признаков
                для всех примеров. При этом мы сохраняем отображение между сортированными
                и изначальными индексами, чтобы знать соответствие между x, y и w

                indices[i] == изначальный индекс элемента, i-го в порядке сортировки
            
            На выходе:
            
            численное значение ошибки обученного классификатора
        '''
        w = np.take(w, indices)
        y = np.take(y, indices)
            
        
        # Какой ужас!
        # Так и хочется переписать на Си!
        def _learn(X, y, w):
            
            s1 = y*w
            s1[1:] = s1[:-1]
            s1[0] = 0
            s1 = np.cumsum(s1)

            y = (y == 0).astype(y.dtype)
            s2 = y*w
            s2 = np.flipud(s2)
            s2 = np.cumsum(s2)
            s2 = np.flipud(s2)
            
            error = s1 + s2 
            
            n = np.argmin(error)
            
            return X[n], error[n]
        # Ужас!

        x_pos, e_pos = _learn(X, y, w)
        
        X = np.flipud(X)
        y = np.flipud(y)
        w = np.flipud(w)
        
        x_neg, e_neg = _learn(X, y, w)

        if e_pos <= e_neg:
            self.threshold = x_pos
            self.polarity  = 1
            error = e_pos
        else:
            self.threshold = x_neg
            self.polarity  = -1
            error = e_neg
            
        return error
                
    def classify(self, x):
        return np.array(self.polarity * x >= self.polarity * self.threshold).astype('int')
        #return 1 if self.polarity * x >= self.polarity * self.threshold else 0
    
    def __repr__(self):
        return "Threshold: {}, polarity: {}".format(self.threshold, self.polarity)

#==============================================================================
# Бустинговый классификатор

class BoostingClassifier:
    def __init__(self, classifiers, weights, ftr_idxs, threshold = None):
        self.classifiers = classifiers
        self.weights = weights
        self.ftr_idxs = ftr_idxs
        self.threshold = sum(weights) / 2 if threshold is None else threshold
    
    def classify(self, X, ret_qa = False):
        '''
        На входе:
        X -- одномерный numpy вектор признаков

        На выходе:
        1, если ансамбль выдает значение больше threshold и 0 если меньше
        '''
        res = 0.0
        for classifier, weight, ftr_idx in zip(self.classifiers, self.weights, self.ftr_idxs):
            res += weight * classifier.classify(X[ftr_idx])
            
        ret_val = int(res > self.threshold)
            
        if ret_qa:
            return ret_val, res/self.threshold
        else:
            return ret_val

#==============================================================================
# Обучение методом бустинга
class ViolaJonesСlassifier(object):
    def __init__(self, rounds = 200, eps = 1e-15):
        self.rounds = rounds
        self.eps    = eps
        self.cls    = None #Классификатор
        self.ftrs   = None #Набор фичей
        

    def train_classifier(classifier_type, X, y, w, indices):
        classifier = classifier_type()
        error = classifier.train(X, y, w, indices)
        return error, classifier

    def learn_best_classifier(classifier_type, X, y, w, indices):
        '''
        Функция находит лучший слабый классификатор
        
        На входе:
        classifier_type -- класс классификатора (DecisionStump в нашем случае)
        X -- двумерный numpy массив, где X[i, j] -- значение признака i для примера j
        Каждый X[i] отсортирован по возрастанию
        y -- одномерный numpy массив с классом объекта (0|1). Порядок y соответствует порядку примеров в датасете
        w -- одномерный numpy массив весов для каждого примера. Порядок w соответствует порядку примеров в датасете
        indices -- список одномерных numpy массивов. 
        indices[i, j] == изначальный индекс элемента, j-го в порядке сортировки для i-го признака
            
        На выходе:
        best_classifier -- лучший слабый классификатор
        best_error -- его ошибка
        best_feature -- признак, на котором он был обучен (одна из HaarFeatures)
        predictions -- предсказания классификатора (в порядке до сортировки)
        '''    
        # натренируем каждый классификатор по каждому признаку
        errors  = []
        classes = []
        N = X.shape[1]
            
        bar = progressbar.ProgressBar()
    
        for i in bar(range(0, N)):
            
            err, cls = ViolaJonesСlassifier.train_classifier(classifier_type, X[i,:], y, w, indices[i,:])
            # Добавляем в списки
            errors.append(err)
            classes.append(cls)
    
        # выберем наилучший и сохраним лучший классификатор, ошибку, признак и индекс признака в 
        # best_classifier, best_error, best_feature, best_feature_ind
        # Как то так:
        # https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
        i = np.array(errors).argmin()
    
        best_feature_ind = i
        best_error       = errors[i]
        best_classifier  = classes[i]
    
        # вернем также предсказания лучшего классификатора
        predictions = np.zeros(len(y))
        for j in range(0, len(y)):
            predictions[indices[best_feature_ind][j]] = best_classifier.classify(X[best_feature_ind][j])
        
        return best_classifier, best_error, best_feature_ind, predictions
            
    def fit(self, X, y):
        '''
        На входе:
            X -- двумерный numpy массив, X[i,j] == значение признака j для примера i
            y -- одномерный numpy массив с классом объекта (0|1)
            rounds -- максимальное количество раундов обучения
            eps -- критерий останова (алгоритм останавливается, если новый классификатор имеет ошибку меньше eps)

        На выходе:
            классификатор типа BoostingClassifier
        '''
        # Транспонируем матрицу пример-признак к матрицу признак-примеры
        print('Transpose X...')
        X_t = X.copy().T
        indices = np.zeros(X_t.shape).astype(int)
        print('Done!\nSort X[i]...')
        # Предсортируем каждый признак, но сохраним соответствие между индексами
        # в массиве indices для каждого прзинака
        bar = progressbar.ProgressBar()
        for index in bar(range(0, len(X_t))):
            indices[index] = X_t[index].argsort()
            X_t[index].sort()
            
        print('Done!\nInitiate learning procedure...')
        # найдем количество положительных примеров в выборке
        n_positive = np.sum(y.astype('int'))
        # найдем количество отрицательных примеров в выборке
        n_negative = len(y) - n_positive
        # инициализируем веса
        w = (1.0 / float(n_positive)) * y.astype('float') + (1.0 / float(n_negative)) * (y == 0).astype('float')
        print('Done!\nWill train the classifier...')
        classifiers = []
        ftr_idxs = []
        alpha = []
        for round in range(0, self.rounds):
            print("Раунд {}".format(round))
            # нормируем веса так, чтобы сумма была равна 1
            w /= np.sum(w)
            # найдём лучший слабый классификатор
            weak_classifier, error, ftr_idx, weak_classifier_predictions = ViolaJonesСlassifier.learn_best_classifier(DecisionStump, X_t, y, w, indices)
            print("Взвешенная ошибка текущего слабого классификатора: {}".format(error))
            # если ошибка уже почти нулевая, остановимся
            if error < self.eps:
                break
            
            # найдем beta
            beta = error / (1.0 - error)
            # e[i] == 0 если классификация правильная и 1 наоборот
            e  = (y != weak_classifier_predictions).astype('float')
            ne = 1.0 - e
            # каждый правильно классифицированный вес нужно домножить на beta 
            w *= (e + beta*ne)#np.power(beta, 1.0 - e)
            # добавим к ансамблю новый классификатор с его весом и признаком
            classifiers.append(weak_classifier)
            ftr_idxs.append(ftr_idx)
            alpha.append(math.log(1.0 / beta))
            
            # посчитаем промежуточную точность
            strong_classifier = BoostingClassifier(classifiers, alpha, ftr_idxs)
            predictions = np.array([strong_classifier.classify(X[i]) for i in range(0, len(X))])
            
            pos_predictions = np.sum((predictions * y).astype('float'))
            neg_predictions = np.sum((predictions * (1 - y)).astype('float'))
            
            correct_positives = pos_predictions / n_positive
            correct_negatives = 1.0 - neg_predictions / n_negative
            
            print("Correct detected faces {}".format(correct_positives))
            print("Correct detected non-faces {}".format(correct_negatives))
            
        print('Done!')
        
        self.cls = BoostingClassifier(classifiers, alpha, ftr_idxs)
        
    def add_features(self, features):
        '''
        Вызываем после fit, на входе - набор фичей, которые 
        применялись к датасету при подготовке к обучению.
        
        Функция вбирает использованные ври обучении классификатора фичи 
        и запоминает их.
        '''        
        self.ftrs = [features[i] for i in self.cls.ftr_idxs]
        self.cls.ftr_idxs = list(range(0,len(self.ftrs)))
        
    def classify_win(self, window, ret_qa = False):
        return self.cls.classify(compute_features_for_image(IntegralImage(window), self.ftrs), ret_qa)

    def classify_wlist(self, wlist, ret_qa = False):
        return [self.classify_win(win, ret_qa) for win in wlist]
        
    def calibrate(self, img_pos, img_neg, rate = 0.5, N = 20):
            
        pivot_thr = 0.5*sum(self.cls.weights)

        thr     = []
        fls_pos = []
        
        bar = progressbar.ProgressBar()
        for i in bar(range(0, N + 1)):
            
            # Не хочется делать полный брутфорс
            cur_thr = pivot_thr * (1 + rate * (i / N - 0.5))
            
            self.cls.threshold = cur_thr
            
            pred_pos = self.classify_wlist(img_pos)
            detection_rate = sum(pred_pos) / len(pred_pos)
      
            if detection_rate > 0.9:
                pred_neg = self.classify_wlist(img_neg)
                false_positive_rate = sum(pred_neg) / len(pred_neg)
                fls_pos.append(false_positive_rate)
                thr.append(cur_thr)
        
        i = np.array(fls_pos).argmin()
        print("False positive rate(%): {}".format(fls_pos[i] * 100))

        # В конце установить подходящее значение порога
        self.cls.threshold = thr[i]
        
    def detect_multi(self, image, img_sz, step = 1):
        norm_image = normalize_image(image)
        w, h = norm_image.shape
        # лучше задавать не абсолютные размеры окна, а относительные (в процентах)
        window_sizes = [0.1, 0.2, 0.4, 0.8]
        results = []
        for w_size in window_sizes:
            
            res_scaled = []
            bar = progressbar.ProgressBar()
            for x in bar(range(0, w, step)):
                for y in range(0, h, step):
                    xc = x + int(h * w_size)
                    yc = y + int(h * 2/3 * w_size) # - пропорции лица по ширине/высоте
                    if xc < w and yc < h:
                        crop = norm_image[x:xc,y:yc]
                        # здесь необходимо нормализовать изображение и применить классификатор
                        # если классификатор детектирует лицо, нужно добавить (x, y, xc, yc) к списку result
                        crop_resized = resize(crop, (img_sz, img_sz), mode='constant').astype(np.float32)
                        is_face, face_qa = self.classify_win(crop_resized, ret_qa = True)
                        #
                        if is_face:
                            res_scaled.append((x, y, xc, yc, face_qa))
                            
            results.append(res_scaled)
        #
        return results
    
    def detect(self, image, img_sz, step = 2):
        
        ret = []
        for res in self.detect_multi(image, img_sz, step):
            ret += res
        
        return ret
