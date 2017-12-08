#!/usr/bin/env python3
# coding: utf-8

"""
FLAMENet flame detector.
Copyright (c) 2017 Kirill Sobyanin ()
Copyright (c) 2017 Alexey Trofimov ()
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
import os

import numpy as np

from sklearn.svm import LinearSVC

from keras.models import Model
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.regularizers import l2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.imagenet_utils import preprocess_input


import warnings
warnings.filterwarnings("ignore")

class FLAMENet(object):
    def __init__(self, C = 0.1, max_iter = 1000, tol = 0.0001, pool_sz = 8):
        self.C        = C
        self.max_iter = max_iter
        self.tol      = tol
        self.model    = InceptionResNetV2(include_top=False, weights='imagenet', pooling="avg") # Или может сделать InceptionResNetV2(include_top=False, weights='imagenet') ?
        self.cls      = LinearSVC(C = self.C, tol = self.tol, max_iter = self.max_iter)
        self.trained_ = False
        self.ready    = False
        self.pool_sz  = pool_sz
        
    def get_features(self, img_folder, size=128, resize=False, new_size=(299, 299), gpu_batch_size=8, feature_count=1536):
        '''
        Извлечение фичей базовой моделью.
        
        files: список файлов для извлечения фичей
        size: количество единовременно считываемых в память файлов
        resize: нужно ли делать ресайз картинки (вообще он обязательно нужен для подсчета фичей 
                                                 пачками, но можно отключить для ускорения, 
                                                 если файлы предварительно нарезаны 
                                                 на одинаковый размер)
        new_size: размер картинки после ресайза, релевантен только для resize=True
        gpu_batch_size: количество файлов, отправляемых для подсчета на GPU (зависит от количества памяти)
        feature_count: количество фичей (зависит от архитектуры сети, 1536 для IRNv2)
        '''
        
        if self.ready:
            return None
        
        def get_batches(files, size=size):
            return (files[pos:pos + size] for pos in range(0, len(files), size)) #получаем генератор по списку seq

        def read_batch(files, mode="tf", resize=resize, new_size=new_size):
            batch = []
            
            for file in files:
                
                image = Image.open(file)
                
                if resize:
                    image = image.resize(new_size, Image.ANTIALIAS)
                    
                batch.append(np.asarray(image))
            return preprocess_input(np.asarray(batch, dtype=np.float64), mode=mode)

        features = []
        files = [os.path.join(img_folder, f) for f in os.listdir(img_folder)]
        batches = list(get_batches(files))
        
        for batch in batches:
            img_data = read_batch(batch, mode="tf")
            raw_features = self.model.predict(img_data, batch_size=gpu_batch_size)
            features.extend(raw_features.reshape(len(batch), feature_count))
        
        return features
        
    def fit(self, X, y, sample_weight=None):
        self.cls.fit(X, y, sample_weight)
        self.trained_ = True
        return self
        
    def predict(self, X):
        return self.cls.predict(X)
    
    def predict_proba(self, X):
        return 1.0/(1.0 + np.exp(-self.cls.decision_function(X)))
    
    def score(self, X, y, sample_weight=None):
        return self.cls.score(X,y,sample_weight)
    
    def prepare_model(self):
        '''Тут делаем нейрохирургию...'''
        # Берем базовую модель
        #Убираем лишнее
        base_model = Model(input=self.model.input, output=self.model.get_layer('conv_7b_ac').output)
        #Добавляем нужные нам слои
        x = AveragePooling2D((self.pool_sz, self.pool_sz), strides = (1,1), name='final_pool')(base_model.output)
        x = Conv2D(1, (1, 1), activation='sigmoid', name='conv_svm', kernel_regularizer = l2(self.C))(x)
        #Делаем финальную модель
        final_model = Model(input=base_model.input, output=x)
        final_model.compile(loss='hinge', optimizer='adadelta', metrics=['accuracy'])
        
        '''Задаем веса руками, если у нас есть веса...'''
        if self.trained_:
            
            Wsvm,Bsvm = final_model.layers[-1].get_weights()
        
            Wsvm[0,0,:,0] = self.cls.coef_[0,:].astype(Wsvm.dtype)
            Bsvm          = self.cls.intercept_.astype(Bsvm.dtype)

            final_model.layers[-1].set_weights([Wsvm, Bsvm])
        
        '''Модель готова, можно пользовать'''
        self.model = final_model
        self.ready = True
        
    def detect_(self, img, thr = 0.75):
        
        pp_img = preprocess_input(np.asarray(img, dtype=np.float64), mode='tf')
        pp_img = np.expand_dims(pp_img, axis=0)

        result = self.model.predict(pp_img)[0,:,:,0]
        #TODO: Перенести в выходной слой!
        #result = 1.0/(1.0 + np.exp(-result))
        
        return (result > thr)
    
    def detect(self, img, thr = 0.75):
        #TODO: Этот метод надо проверить на предмет правильности боксов
        if not self.ready:
            return None
        
        result = self.detect_(img, thr)
        
        w,h = result.shape
        boxes = []
        for i in range(0, w):
            for j in range(0, h):
                if result[i,j]:
                    x = (j*100)//3
                    y = (i*100)//3
                    d = ((self.pool_sz + 1)*100)//3 - 1
                    boxes.append([(x, y), (x + d, y + d)])
                    
        return boxes, result.astype('int')


if __name__ == '__main__':
    #import shutil as sh
    import sys
    import h5py
    import cv2
    from sklearn import metrics as mt
    from sklearn.model_selection import StratifiedKFold
    
    from PIL import Image, ImageDraw
    

    #import warnings
    #warnings.filterwarnings("ignore")
    '''
    with h5py.File('smoke_features.h5', 'r') as hf:
        smoke_features = hf['InceptionResNetV2'][:]
        
        with open('smoke_files.txt', 'r') as f:
            smoke_files = f.read().splitlines()
    '''    
    with h5py.File('fire_features.h5', 'r') as hf:
        fire_features = hf['InceptionResNetV2'][:]
    
    with open('fire_files.txt', 'r') as f:
        fire_files = f.read().splitlines()
    
    with h5py.File('negative_features.h5', 'r') as hf:
        negative_features = hf['InceptionResNetV2'][:]
    
    with open('negative_files.txt', 'r') as f:
        negative_files = f.read().splitlines()

    X = np.vstack([fire_features, negative_features])
    y = np.concatenate([np.ones(len(fire_features)), np.zeros(len(negative_features))])
    files = fire_files + negative_files

    np.random.seed(228228)
    ind = list(range(len(X)))
    np.random.shuffle(ind)
    
    X = X[ind]
    y = y[ind]
    
    cls = FLAMENet()
    
# =============================================================================
## Тест фичей:
#     folder = "D:\\notebooks\\CV School\\project\\Inception features SMAL\\test"
#     fs = cls.get_features(folder, resize=True)
# =============================================================================
    
    '''
    kf = StratifiedKFold(n_splits=5, random_state=1337)

    tr,ts = next(iter(kf.split(X,y)))
    
    cls.fit(X[tr], y[tr])
    
    y_ts = y[ts]
    
    print("Will predict X...")
    y_pred = cls.predict(X[ts])
    print('Precisision:', mt.precision_score(y_ts, y_pred), 'Recall:', mt.recall_score(y_ts, y_pred))
    '''
    print('Will train LinearSVC')
    cls.fit(X, y)
    #Делаем модель
    print('Done!\nWill prepare model...')
    cls.prepare_model()
    print('Done!')


    #==========================================================================
    def process_image(img_path, thr = 0.8):
        global cls
        img = Image.open(img_path)
        
        boxes, result = cls.detect(img, thr)
        im_draw = ImageDraw.Draw(img)
        for box in boxes:
            im_draw.rectangle(box, outline = 'red')
            
        img.show()
        
        print(result.shape)
        print(boxes)
    #==========================================================================
    def process_frame(img, thr = 0.8):
        global cls
        
        boxes, result = cls.detect(img[...,::-1], thr)
        
        for box in boxes:
            cv2.rectangle(img, box[0], box[1], (0,0,255), 2)
            
        return img
    #==========================================================================
    def get_boxes(img, thr = 0.8):
        global cls
        
        boxes, result = cls.detect(img[...,::-1], thr)
            
        return boxes
    #==========================================================================    
    
    input_file  = '0.avi'
    output_file = 'out_0.avi'
    
    # Количество кадров для обновления рамки огня + инициализируем рамки
    N_FRAMES = 20
    boxes = [] 
    
    #Open input file
    cap = cv2.VideoCapture(input_file)
    #cap = cv2.VideoCapture(0) # раскомментить для вебкамеры 
    
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    #==========================================================================
    #Open output file cv2.VideoWriter_fourcc(*'MJPG)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_file, fourcc, 25.0, (cap_w, cap_h))
    
    fgbg = cv2.createBackgroundSubtractorMOG2()

    #Now process the file
    cv2.namedWindow("frame")
    
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if not ret:
            break

        fgmask = fgbg.apply(frame)

        #if cap.get(cv2.CAP_PROP_POS_FRAMES) % N_FRAMES:   # Работает только для видеофайлов (не работает с видеопотоком камеры)
        if i % N_FRAMES:
            # Этот If срабатывает N-1 раз из N кадров. Тут записываем текущий фрейм + рамки с последнего
            # Проанализированного кадра
            
            for box in boxes:
                cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 2)
            
            out.write(frame)
            cv2.imshow('frame', frame)
            i += 1
            continue
        
        print('---------------')
        
        fgmask = cv2.GaussianBlur(fgmask, (11,11), 0)
        fgmask = (fgmask > 127).astype('float')
        fgmask = cv2.GaussianBlur(fgmask, (11,11), 0)
        mframe = frame#.copy()
        mframe[:,:,0] *= fgmask
        mframe[:,:,1] *= fgmask
        mframe[:,:,2] *= fgmask
        boxes = get_boxes(mframe, 0.5)
        
        for box in boxes:
            cv2.rectangle(frame, box[0], box[1], (0,0,255), 2)

        out.write(frame)
        
        cv2.imshow('frame', frame)
        i += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        

