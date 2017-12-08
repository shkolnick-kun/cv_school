#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:03:24 2017

@author: anon
"""
import os
import shutil as sh

import progressbar

import pickle

import h5py
import numpy as np


from sklearn import metrics as mt
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold



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

'''
X = np.vstack([fire_features, negative_features])
y = np.concatenate([np.ones(len(fire_features)), np.zeros(len(negative_features))])
files = fire_files + negative_files
'''
'''
NOTE: Тут будет эксперимент по преобразованию датасета для последующей 
классификации последователности кадров, попробуй SVM-мом разделить 
последовательности с огнем и без огня...

Ввиду того, что негатив готовился просто по случайным кадрам, 
попробую отличать огонь от дыма
'''
def fname_parse(fname):
    a,b = fname.split('][')
    src,c = a.split('[')
    d,dst = b.split(']')
    x1,y1 = c.split()
    x2,y2 = d.split()
    frmid = int(dst.split('.')[0])
    return [src,x1,y1,x2,y2], frmid

def process_files_and_features(features, files, N = 5):
    features = list(features)
    
    timed_features = []
    timed_files    = []
    
    file_id     = []
    frm_id      = None
    
    feature_win = []

    i = 0
    bar = progressbar.ProgressBar(maxval = len(files))
    bar.start()
    
    for fname, vec in zip(files, features):
        i += 1
        bar.update(i)
        #Ищем начало клипа, складываем данные в FIFO
        new_file, new_frm = fname_parse(fname)
        if (file_id == new_file) and (new_frm - frm_id == 10):
            #Продолжается старый клип
            feature_win.append(vec)
            frm_id = new_frm
        else:
            #Начало нового клипа
            file_id = new_file
            frm_id  = new_frm
            feature_win = [vec]
            continue

        if len(feature_win) > N:
            #Длина FIFO должна быть N
            feature_win.pop(0)
            
        if len(feature_win) == N:
            #Формируем вектор фич и готовим к выводу
            timed_vec = np.array(feature_win).flatten()
            timed_features.append(timed_vec)
            timed_files.append(fname)
        
    return (np.array(timed_features), timed_files)
      
'''
test_files = ['smoke_smal/01.mp4[1405    0][1918  513]1170.0.jpg',
              'smoke_smal/01.mp4[1405    0][1918  513]1180.0.jpg',
              'smoke_smal/01.mp4[1405    0][1918  513]1190.0.jpg',
              'smoke_smal/01.mp4[1405    0][1918  513]1200.0.jpg',
              'smoke_smal/01.mp4[1405    0][1918  513]1210.0.jpg',
              'smoke_smal/01.mp4[1405    0][1918  513]1220.0.jpg',
              'smoke_smal/01.mp4[1405    0][1918  513]1230.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1100.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1110.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1120.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1130.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1140.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1100.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1110.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1120.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1130.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1140.0.jpg',
              'smoke_smal/02.mp4[1405    0][1918  513]1150.0.jpg',]

test_features = np.array([[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],
                          [2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],
                          [3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7]])

print(process_files_and_features(test_features, test_files))
'''
def get_timed_data(cachefile, features, files):
    
    if os.path.isfile(cachefile):
        return pickle.load(open(cachefile, 'rb'))
    else:
        ret = process_files_and_features(features, files)
        pickle.dump(ret, open(cachefile, 'wb'))
        return ret

print('Will get dataset...')
fire_features,  fire_files  = get_timed_data('timed_fire.pkl',   fire_features,  fire_files)
negative_features, negative_files = get_timed_data('timed_neg.pkl', negative_features, negative_files)
print('Done!')

def compute_dynamic_features(vec):
    m_shape = (5, 1536)
    mat = np.reshape(vec, m_shape)
    ret = np.zeros(m_shape, mat.dtype)
    #Структура не нужна
    #ret[0,:] = mat[2,:]
    #Динамика
    ret[1,:] = 0.5*(          mat[3,:] -                    mat[1,:]          )
    ret[2,:] =                mat[3,:] - 2.0*mat[2,:] +     mat[1,:]
    ret[3,:] = mat[4,:] - 2.0*mat[3,:]                + 2.0*mat[1,:] - mat[0,:]
    ret[4,:] = mat[4,:] - 4.0*mat[3,:] + 6.0*mat[2,:] - 4.0*mat[1,:] + mat[0,:]
    ret = ret.flatten()
    #Relu
    ret *= (ret > 0).astype(mat.dtype)
    return ret

def transform_features(features):
    for i in range(0, len(features)):
        features[i] = compute_dynamic_features(features[i])
    return features
       
print('Will transform features...') 
fire_features = transform_features(fire_features)
negative_features = transform_features(negative_features)
print('Done!')

print(fire_features.shape, negative_features.shape)

X = np.vstack([fire_features, negative_features])
y = np.concatenate([np.ones(len(fire_features)), np.zeros(len(negative_features))])
files = fire_files + negative_files

np.random.seed(228228)
ind = list(range(len(X)))
np.random.shuffle(ind)

X = X[ind]
y = y[ind]

'''
cls = LinearSVC(C=0.1)
print('Will train SVC')
cls.fit(X, y)
pickle.dump(cls, open('TimedSVC.pkl', 'wb'))
print('Done!')
'''

kf = StratifiedKFold(n_splits=5, random_state=1337)
for tr,ts in iter(kf.split(X,y)):
    print("Will train linSVC2...")
    cls = LinearSVC(C=0.1)#SVC(C=0.1, kernel = 'linear', max_iter = 100)
    cls.fit(X[tr], y[tr])
    
    y_ts = y[ts]
    
    print("Will predict X...")
    y_pred = cls.predict(X[ts])
    #print(cls.n_support_)
    print('Precisision:', mt.precision_score(y_ts, y_pred), 'Recall:', mt.recall_score(y_ts, y_pred))
    
    print("Will predict X probability...")
    p_pred = 1.0/(1.0 + np.exp(-cls.decision_function(X[ts])))
    print('Done!')
    
    tp = y_pred * y_ts
    tn = (1.0 - y_pred)*(1.0 - y_ts)

    fp = y_pred - tp
    fn = (1.0 - y_pred) - tn

    idx  = np.array(range(0, len(y_ts)))
    #i_fn = list(ind[fn.astype('bool')])
    #i_fp = list(ind[fp.astype('bool')])
    
    print('False positive files:')
    for i in list(idx[fp.astype('bool')]):
        fl_id = ind[ts[i]]
        print(files[fl_id])
        print('Probability:', p_pred[i])
        sh.copy(files[fl_id], 'false/TimedLinSVC/pos')
        #im = io.imread(files[fl_id])
        #io.imshow(im)

    print('False negative files:')   
    for i in list(idx[fn.astype('bool')]):
        fl_id = ind[ts[i]]
        print(files[fl_id])
        print('Probability:', p_pred[i])
        sh.copy(files[fl_id], 'false/TimedLinSVC/neg')
        #im = io.imread(files[fl_id])
        #io.imshow(im)
