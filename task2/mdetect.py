#!/usr/bin/env python
# -*- coding: utf-8 -*-s
"""**************************************************************************
    Motion sensor
    Copyright (C) 2017 Paul Beltyukov
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    Please contact with me by E-mail: beltyukov.p.a@gmail.com
**************************************************************************"""
import sys
import os
import cv2
import numpy as np
#В OpenCV 2.4 нет функций для разметки связянных областей
from skimage import measure as skim
#==============================================================================
#                             Motion detector!
#==============================================================================
def get_grad(img,ksize):
    Gx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize)
    Gy = cv2.Sobel(img,cv2.CV_32F,0,1,ksize)
    return np.sqrt(Gx*Gx+Gy*Gy)
    
def noice_cancel(img,k_thr,ksize):
    mx = np.amax(img)
    r,nc = cv2.threshold(img, mx * k_thr, 1.0, cv2.THRESH_TOZERO)
    #Смягчение границ
    return cv2.GaussianBlur(nc,(ksize,ksize),0)

class flash_auto():
    def __init__(self, vthr, cthr, istate, alpha):
        self.val_z = 0
        self.vthr  = vthr
        self.cnt   = 1.0
        self.cthr  = cthr
        self.state = istate
        self.alpha = alpha
   
    def run(self, val):
        if self.val_z > 0 and self.val_z * self.vthr < val:
            self.state  = True
            self.cnt    = 1.0
        
        if self.state:
            if self.cnt > self.cthr:
                self.cnt = self.cnt * (1.0 - self.alpha)
            else:
                self.state = False
        
        self.val_z = val

        return self.state
        

class BgEstimator():
    def __init__(self, shape, k_alf = 0.1, alpha = 0.1):

        if len(shape) != 2:
            print 'BgEstimator: shape must have form (x,y) !'

        self.avg_frame = np.zeros(shape, 'float32')
        self.avg_noise = np.zeros(shape, 'float32')
        self.k_alf = k_alf
        self.alpha = alpha
        #Отработка вспышек
        self.flash = flash_auto(10.0, 0.000001, True,  alpha) #TODO:Подобрать эмпирически, или написать самонасройку
        #Отработка опорных кадров
        self.ifrm  = flash_auto(1.2, 0.4,      False, alpha)  #TODO:Подобрать эмпирически, или написать самонасройку
        #Фильтруем выход
        self.ng_ksz = 41
        self.k_nr   = 1.0
        #Служебные данные
        self.nsden_z = 0

    def _run(self, img):
        fast = self.alpha
        mid  = self.k_alf * fast
        slow = self.k_alf * mid
   
        #Отклонение от фона
        diff = cv2.absdiff(img, self.avg_frame)

        #Вычисление масок
        pos_msk = cv2.compare(diff, self.avg_noise, cv2.CMP_LE)
        neg_msk = cv2.bitwise_not(pos_msk)

        #Оценка шума
        cv2.accumulateWeighted(diff, self.avg_noise, slow, mask = pos_msk)
        cv2.accumulateWeighted(diff, self.avg_noise,  mid, mask = neg_msk)

        #Оценка фона
        cv2.accumulateWeighted(img, self.avg_frame, fast, mask = pos_msk)
        cv2.accumulateWeighted(img, self.avg_frame,  mid, mask = neg_msk)

        return neg_msk, diff
    
    def _compute_blured(self, img, ksz):
        kshape = (ksz, ksz)
        #
        i   = cv2.GaussianBlur(img, kshape, 0)
        avi = cv2.GaussianBlur(self.avg_frame, kshape, 0)
        #
        d   = cv2.absdiff(i, avi)
        #
        thr = np.average(self.avg_noise) * self.k_nr
        n   = cv2.compare(d, thr, cv2.CMP_GT)
        #
        return n,d
        

    def run(self, img):

        #Оцениваем фон
        neg_msk, diff = self._run(img)

        # Фильтруем
        # Для переделки в "неколько градиентов" 
        # достаточно вызвать self._compute_blured 
        # с несколькими размерами ядра и
        # объединить n, например, - "по или"
        neg,d = self._compute_blured(img, self.ng_ksz)

        # Т.к. шумы оцениваются "не совсем сверху",
        # считаем плотность "шумовых" областей движения.
        pos = cv2.compare(neg, 0, cv2.CMP_EQ)
        pos = cv2.bitwise_and(pos, 1)
        # Единицы там, где может быть "шумовая" составляющая
        ns     = cv2.bitwise_and(neg_msk, pos)
        nsden  = np.sum(ns).astype('float')/np.sum(pos)

        #Вспышка - резкий рост плотности
        flash = self.flash.run(nsden) or self.ifrm.run(nsden)

        #Обновление служебной информации
        print "Dencity change:", nsden/self.nsden_z
        self.nsden_z    = nsden

        return neg, flash, d


class MotionSensor(BgEstimator):
    def __init__(self, shape, k_alf = 0.1, alpha = 0.1, bl_ksz = 7):

        #
        BgEstimator.__init__(self, shape, k_alf, alpha)

        # Ядро для морфологических операций
        self.krn = np.ones((bl_ksz, bl_ksz), np.float32)


    def run(self, img):
        #=========================================================
        # Алгоритм оценки фона получился такой,                  #
        # что делать блур до вычисления фона бесполезно,         #
        # маска на выходе от этого не меняется!!!                #
        #=========================================================

        # Считаем градиент
        g = get_grad(l, 3)

        # Выделяем фон
        msk, flash, diff = BgEstimator.run(self, g)

        # Формируем "правильные окна" для объектов
        msk = cv2.dilate(msk, self.krn, iterations = 2)
        msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, self.krn*2)
        
        # Разметка областей
        msk = skim.label(msk.astype('int'), neighbors = 8)

        # Анализ областей, для анализа в дальнейшем можно будет использовать градиенты
        regions = skim.regionprops(msk, g)

        # Фильтруем области
        flt_msk = np.zeros(msk.shape, 'int')
        flt_reg = []
        i = 1
        for reg in regions:
            # смотрим маску
            x1,y1,x2,y2 = reg.bbox
            cur_msk = np.zeros(msk.shape, 'int')
            cur_msk[x1:x2, y1:y2] = reg.filled_image
            # Пропускаем только то, что не ноль, бывали случаи артефактов
            if np.sum(cur_msk * msk) != 0:
                flt_msk = cv2.bitwise_or(flt_msk, cur_msk * i)
                reg.label = i
                flt_reg.append(reg)
            else:
                print 'Art:' + str(i)
            #
            i = i + 1
        
        return flt_msk, len(flt_reg), flash, diff

#==============================================================================
#                                  Main
#==============================================================================
if __name__ == '__main__':

    if len(sys.argv) < 2:
        input_file  = 0
        output_file = 'output.avi'
    else:
        input_file = sys.argv[1]

        if not os.path.isfile(input_file):
            print 'Wrong filename!'
            sys.exit(-1)

        out_dir, out_basename = os.path.split(os.path.realpath(input_file))
        out_basename = out_basename[:-4] + '.avi'
        output_file = os.path.join(out_dir, 'out_' + out_basename)
        print output_file

    #Open input file
    cap = cv2.VideoCapture(input_file)

    cap_w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))  # float
    cap_h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float

    #==============================================================================
    #TODO: Перенести инициацию в конструктор
    #==============================================================================
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    m_sense = MotionSensor((cap_h, cap_w), k_alf = 0.1, alpha = 0.2)

    #==============================================================================
    #Open output file
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (2*cap_w, 2*cap_h))

    big_frame = np.zeros((2*cap_h, 2*cap_w, 3),'uint8')

    #Now process the file
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # Should use LAB as L is beter than V
            l,a,b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
            
            """
            =============================================================================
            Детектор движения
            =============================================================================
            """
            #Эквализация
            l = clahe.apply(l)
            # Детектор движения
            b,rn,f,a = m_sense.run(l)
            """
            =============================================================================
            Обработка резултатов
            =============================================================================
            """
            g = np.ones(l.shape,'uint8')
            if f:
                g = g*255
            else:
                g = 0

            if rn > 0:
                b = (b.astype('float') * 255.0/rn).astype('uint8')

            l = cv2.convertScaleAbs(l, alpha=1.0, beta=0.0)
            a = cv2.convertScaleAbs(a, alpha=1.0, beta=0.0)
            """
            =============================================================================
            Вывод результатов
            =============================================================================
            """

            big_frame[0:cap_h,            0:cap_w,:] = frame; #Left upper
            #Rigth upper
            big_frame[0:cap_h,      cap_w:2*cap_w,0] = l
            big_frame[0:cap_h,      cap_w:2*cap_w,1] = g
            big_frame[0:cap_h,      cap_w:2*cap_w,2] = b      
            #Rigth lower
            big_frame[cap_h:2*cap_h,cap_w:2*cap_w,0] = a      
            #Left lower
            big_frame[cap_h:2*cap_h,      0:cap_w,0] = b      
            # write the flipped frame
            out.write(big_frame)

            cv2.imshow('frame',big_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
