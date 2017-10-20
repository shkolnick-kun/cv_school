#!/usr/bin/env python
# -*- coding: utf-8 -*-s
import sys
import os
import cv2
import numpy as np
#=======================================================================================================================
#We don't have drawMatches in OpenCV 2.4, so we should use this:
#https://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python/26227854#26227854
#if we need to draw them
#=======================================================================================================================
"""**************************************************************************
    Prokudin Gorsky photo assembler
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
#=====================================================================
# Функция взвешивания для сортировки совпадающих фич
def get_match_key(m_n):
    (m,n) = m_n
    return float(m.distance)/(float(n.distance)+1.0)

#=====================================================================
def get_matches(matcher, dsc1, dsc2, N = 5, Q = 0.5):

    matches = matcher.knnMatch(dsc1, dsc2, k=2)

    if len(matches)>0:
        print 'Matches found:', len(matches)
    else:
        print 'No matches found!'
        return []
    
    # Фильтруем совпадения
    flt = []
    for m in matches:
        
        if len(m) != 2:
            continue

        if m[0].distance < Q*m[1].distance:
            flt.append(m)
    
    # Сортируем по качеству
    flt = sorted(flt, key=get_match_key)

    # Отбираем N самых качественных 
    good = []
    for i in range(0, min(N, len(flt))):
        (m,n) = flt[i]
        good.append(m)

    return good

#=====================================================================
def get_ctr_and_vecs(pt):

    pt = np.array(pt)

    # Центр
    c = np.average(pt, axis=0)

    # Радиус-векторы
    v = pt - c

    # Модуль вектора
    m = np.linalg.norm(v, axis=1)

    # Направляющие косинусы
    cs = v
    for i in range(0, m.shape[0]):
        cs[i,:] = v[i,:]/m[i]

    return c, v, cs

#=====================================================================
def get_shift_and_angle(pt1, pt2):
    # Получаем центры, радиус-векторы, направления
    c1, v1, cs1 = get_ctr_and_vecs(pt1)
    c2, v2, cs2 = get_ctr_and_vecs(pt2)
    
    # Считаем угол поворота    
    n = v1.shape[0]
    al = []
    for i in range(0, n):
        d1 = cs1[i,:]
        d2 = cs2[i,:]
        
        if np.linalg.norm(d1 - d2) <= 0.0000000000001:
            a = 0
        else:
            a = np.arccos(np.dot(d1, d2))

        al.append(a)

    angle = np.median(np.array(al))

    return c1, c2, v1, v2, c1 - c2, angle

#=====================================================================
def get_matched_points(matcher, kp1, dsc1, pt1, kp2, dsc2, pt2, N = 25, Q = 0.5):
    
    # Получаем совпадающие фичи
    matches = get_matches(matcher, dsc1, dsc2, N, Q)

    # Сдвиг и поворот можно посчитать не меньше, чем для двух точек
    if len(matches) < 2:
        return pt1, pt2

    for mat in matches:
        
        # Берем индексы
        kp1_idx = mat.queryIdx
        kp2_idx = mat.trainIdx

        #Добавляем точки в списки
        pt1.append(kp1[kp1_idx].pt)
        pt2.append(kp2[kp2_idx].pt)

    return pt1, pt2

#=====================================================================
def get_matched_img(kp1, x1, y1, kp2, img2):

    # Рассчитываем сдвиг и поворот
    c1, c2, v1, v2, s, a = get_shift_and_angle(kp1, kp2)

    print s,a

    # Двигаем
    M    = np.float32([[1, 0, s[0]], [0, 1, s[1]]])
    ret = cv2.warpAffine(img2, M, img2.shape)

    # Поворачиваем
    M    = cv2.getRotationMatrix2D((c1[0] + x1, c1[1] + y1), a / np.pi * 180.0, 1.0)
    ret = cv2.warpAffine(ret, M, ret.shape)

    return ret

#=====================================================================
def img_prepare(src):

    x,y = src.shape
    dst_shape = (2*x, 2*y)

    dst = np.zeros(dst_shape, 'uint8')
    dst[:x, :y] = src

    M   = np.float32([[1, 0, x/2], [0, 1, y/2]])
    dst = cv2.warpAffine(dst, M, dst_shape)
    
    return dst, x/2, y/2


#=====================================================================
def get_msk(i, j, M, img):

    hx,hy = img.shape
    msk = np.zeros(img_g.shape, 'uint8')

    hx = hx/M
    hy = hy/M

    msk[hx*i:hx*(i+1),hy*j:hy*(j+1)] = 255

    print hx*i, hx*(i+1), hy*j, hy*(j+1)

    return msk

#=====================================================================
def get_aligned_images(img_r, img_g, img_b, N=5, Q = 0.5, M=5):
    # Не люблю патентованные алгоритмы, использую ORB
    orb = cv2.ORB(scaleFactor = 1.5, edgeThreshold=31, patchSize=31)

    # Испоьзую FLANN, ибо быстрый
    FLANN_INDEX_LSH = 6

    index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,      # 12
                        key_size = 12,         # 20
                        multi_probe_level = 1) # 2    

    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Анализируем картинки, формируем выборки для преобразования img_r, img_b
    c_gr = []
    c_r  = []

    c_gb = []
    c_b  = []

    for i in range(0, M):
        for j in range(0, M):

            kp_r, dsc_r = orb.detectAndCompute(img_r, get_msk(i, j, M, img_r))
            kp_g, dsc_g = orb.detectAndCompute(img_g, get_msk(i, j, M, img_g))
            kp_b, dsc_b = orb.detectAndCompute(img_b, get_msk(i, j, M, img_b))

            if len(kp_r) == 0 or len(kp_g) == 0 or len(kp_b) == 0:
                continue

            c_gr, c_r = get_matched_points(flann, kp_g, dsc_g, c_gr, kp_r, dsc_r, c_r, N, Q)
            c_gb, c_b = get_matched_points(flann, kp_g, dsc_g, c_gb, kp_b, dsc_b, c_b, N, Q)

            print 'Shapes kp:', np.array(c_gr).shape, np.array(c_r).shape, np.array(c_gb).shape, np.array(c_b).shape

    # Расширяем картинки для последующей обработки
    new_r,  x,  y = img_prepare(img_r)
    new_g, xg, yg = img_prepare(img_g)
    new_b,  x,  y = img_prepare(img_b)

    # Обрабатываем картинки
    new_r = get_matched_img(c_gr, xg, yg, c_r, new_r)
    new_b = get_matched_img(c_gb, xg, yg, c_b, new_b)

    return new_r, new_g, new_b

#=====================================================================
"""
Prokudin Gorsky photo assembler skeleton
Copyright (c) 2017 Alexey Yastrebov (yastrebov@macroscop.com)

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

"""Получение цветных фотографий из монохромных диапозитивов Прокудина-Горского.
http://www.loc.gov/pictures/collection/prok/
Использование:
photo.py data_dir result_dir

Аргументы:
data_dir
    Каталог с исходными фотографиями.
result_dir
    Каталог, куда складываются полученные цветные фотографии.
"""
if len(sys.argv) < 3:
    print("missing arguments")
    quit()
    
data_dir = sys.argv[1]
result_dir = sys.argv[2]

#=====================================================================
def load_src_images(dir):    
    data = []
    fns = os.listdir(dir)
    for fn in fns:
        if fn.endswith(".png") or fn.endswith(".bmp") or fn.endswith(".jpg"):
            img = cv2.imread(os.path.join(dir,fn),0)
            data.append((fn, img))
    return data

#=====================================================================
def split_triple_image(img):
    """
    Разделение тройного диапозитива на три монохромных изображения.
    Этот алгоритм можно улучшить, но, в принципе, имеющиеся неточности должны быть устранены алгоритмом совмещения 
    изображений.
    """
    h,w = img.shape
    top_shift = 30
    img2 = img[top_shift:h-1,:] #отрежем сверху 30 пикселей
    
    h,w = img2.shape
    h_part = int(h / 3)
    
    img_b = img2[0:h_part,:]
    img_g = img2[h_part:2*h_part,:]
    img_r = img2[2*h_part:3*h_part,:]
    return img_r, img_g, img_b

#=====================================================================
def merge_images(img_r, img_g, img_b):
    """
    Складывание трех одноканальных изображений в одно цветное. Может быть полезным домножить каждый
    компонент на свой коэффициент для получения более точного цвета.
    """
    img_r = img_r.astype(float)
    img_g = img_g.astype(float)
    img_b = img_b.astype(float) * 0.8 #Чувствительность фотоэмульсии к синему была выше
   
    img = cv2.merge((img_b.astype(float), img_g.astype(float), img_r.astype(float)))
    
    return img

#=====================================================================    
data = load_src_images(data_dir)
for d in data:
    #разделяем тройной диапозитив на три части
    img_r, img_g, img_b = split_triple_image(d[1])
    
    #получаем изображения, геометрически преобразованные в результате совмещещния 
    img_r_al, img_g_al, img_b_al = get_aligned_images(img_r, img_g, img_b)
    
    #складываем каналы для получения цветного изображения
    img = merge_images(img_r_al, img_g_al, img_b_al)
    
    cv2.imwrite(os.path.join(result_dir,d[0]),img)
