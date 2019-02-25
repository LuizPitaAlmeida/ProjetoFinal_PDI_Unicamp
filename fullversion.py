# Importação de bibliotecas OpenCV, MatplotLib e Numpy
import cv2 # biblioteca de visao computacional OpenCV
import numpy as np # biblioteca de manipulacao de vetores e ferramentas matematicas e cientificas Numpy
#import matplotlib as mpl # biblioteca de manipulacao e exibicao de dados Matplotlib
#from matplotlib import pyplot as plt # Pyplot: modulo do Matplotlib para exibir as imagens
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path",help="Place the full path of image and text, for example path_to_files/img.png")
args = parser.parse_args()

# get filename for reference
name = args.path[-10:-5]

# leitura da imagem
img_color = cv2.imread(args.path)
img = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)

# variáveis globais
candidatos = []
minArea = 800 #definição da menor área de uma placa, baseado na menor placa de posível leitura
maxArea = 15000 #definição da maior área, baseado na maior placa encontrada (mais próxima da câmera)

def makeSafeCrop(img,rois):
    crops = []
    for plate in rois:
        ny = plate[1] 
        nx = plate[0] 
        nh = plate[3] 
        nw = plate[2] 
        crop = img[ny:ny+nh,nx:nx+nw]
        crops.append(crop)
    return crops

im = cv2.blur(img,(5,5))
im = cv2.equalizeHist(im)
sobelx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=5) # filtro de sobel eixo x
sobelx = cv2.normalize(sobelx, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
_,xthr = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
if(np.sum(xthr.ravel())/(xthr.shape[0]*xthr.shape[1]) > 127): xthr = 255 - xthr
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(12,2)) # MORPH_RECT devido ao formato da placa 
morphDx = cv2.dilate(xthr,kernel,1)
_,contours, hierarch = cv2.findContours(morphDx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#color = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
#cv2.drawContours(color, contours, -1, (0,0,255), 1)
rois1 = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if(area < maxArea and area > minArea):
        x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(color,(x,y),(x+w,y+h),(255,0,0),3)
        ar = 1.0*h/w
        if(ar >= 0.25 and ar <= 0.45):
            #cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),5)
            rois1.append([x,y,w,h])

for crop in makeSafeCrop(img,rois1):
    candidatos.append(crop)    

count_cand = 0
idx_selectedcand = []
center_std = []
for cand in candidatos:
    if(cand.shape[0]!=0 or cand.shape[1]!=0):
        gaussian = cv2.GaussianBlur(cand, (9,9), 10.0)
        unsharp_image = cv2.addWeighted(cand, 2, gaussian, -0.3, 0)
        ret,thr = cv2.threshold(unsharp_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ker = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        mor = cv2.morphologyEx(thr,cv2.MORPH_CLOSE,ker)
        _,contours,hierarquia = cv2.findContours(mor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #img_cont = cv2.cvtColor(thr,cv2.COLOR_GRAY2BGR)
        foundflag = 0
        center = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            ar = 1.0*h/w
            if(h>=10 and w<=60 and ar > 0.7):
                foundflag = foundflag+1
                center.append(y+h/2)
                #cv2.rectangle(img_cont,(x,y),(x+w,y+h),(0,255,0),1)   
        if foundflag >=5 :
            if(np.std(center) < 5):
                idx_selectedcand.append(count_cand)
                center_std.append(np.std(center))
            #plt.figure()
            #plt.title(str(np.std(center)))
            #plt.imshow(img_cont,'gray')
    count_cand = count_cand + 1

number = len(idx_selectedcand)
if number > 1 :
    placa = candidatos[idx_selectedcand[np.argmin(center_std)]]
    idx = idx_selectedcand[np.argmin(center_std)]
elif number == 1:
    placa = candidatos[idx_selectedcand[0]]
    idx = idx_selectedcand[0]
else:
    count_cand = 0
    idx_selectedcand = []
    center_std = []
    for cand in candidatos:
        if(cand.shape[0]!=0 or cand.shape[1]!=0):
            gaussian = cv2.GaussianBlur(cand, (9,9), 10.0)
            unsharp_image = cv2.addWeighted(cand, 2, gaussian, -0.3, 0)
            ret,thr = cv2.threshold(unsharp_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ker = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
            mor = cv2.morphologyEx(thr,cv2.MORPH_CLOSE,ker)
            _,contours,hierarquia = cv2.findContours(mor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #img_cont = cv2.cvtColor(thr,cv2.COLOR_GRAY2BGR)
            foundflag = 0
            center = []
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                ar = 1.0*h/w
                if(h>=10 and w<=60 and ar > 0.7):
                    foundflag = foundflag+1
                    center.append(y+h/2)
                    #cv2.rectangle(img_cont,(x,y),(x+w,y+h),(0,255,0),1)   
            if foundflag >=2 :
                idx_selectedcand.append(count_cand)
                center_std.append(np.std(center))
        count_cand = count_cand + 1
        if number > 0 :
            placa = candidatos[idx_selectedcand[np.argmin(center_std)]]
            idx = idx_selectedcand[np.argmin(center_std)]
        else:
            placa = candidatos[0]
            idx = 0 


_,thr = cv2.threshold(placa,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_,contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#img_cont = cv2.cvtColor(placa,cv2.COLOR_GRAY2BGR)
#cv2.drawContours(img_cont, contours, -1, (0,255,0), 3)
selected = [0,0,placa.shape[1],placa.shape[0]]
count_cnt = 0
for cnt in contours:
    epsilon = 0.02*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if(len(approx) >= 3):
        x,y,w,h = cv2.boundingRect(cnt)
        if(w*h >= placa.shape[0]*placa.shape[1]/3):
            selected = [x,y,w,h]
            count_cnt+=1
            #cv2.rectangle(img_cont,(x,y),(x+w,y+h),(0,255,0),3)
#plt.imshow(img_cont,cmap='gray'),plt.xticks([]), plt.yticks([])

local = []
local.append([rois1[idx][0]+selected[0],rois1[idx][1]+selected[1],selected[2],selected[3]])

#color = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
#cv2.rectangle(color,(local[0][0],local[0][1]),(local[0][0]+local[0][2],local[0][1]+local[0][3]),(0,255,0),3)
#cv2.namedWindow('teste',cv2.WINDOW_NORMAL)
#cv2.imshow('teste',color)
#cv2.waitKey()
#cv2.destroyAllWindows()

print(name,local[0][0],local[0][1],local[0][2],local[0][3])
