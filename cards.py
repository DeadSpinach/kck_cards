import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageFont


CARD_MIN_AREA = 20000



class Sampleranks:
    def __init__(self):
        self.img = []
        self.name = "Placeholder"

class Samplesuits:
    def __init__(self):
        self.img = []
        self.name = "Placeholder"


def pproc(im):


    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imblur = cv2.GaussianBlur(imgray,(5,5),0)
    img_w, img_h = np.shape(im)[:2]
    threshlvl = 120 + imgray[int(img_h/100)][int(img_w/2)]
    ret, imthresh = cv2.threshold(imblur,threshlvl,255,cv2.THRESH_BINARY)
    cv2.imshow('kck', imthresh)
    return imthresh




def finder(im):
    contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cardlist = np.zeros(len(contours),dtype=int)
    for i in range(len(contours)):
        size = cv2.contourArea(contours[i])
        per = cv2.arcLength(contours[i],True)
        epsilon = per/100
        #print(size)
        approx = cv2.approxPolyDP(contours[i],epsilon,True)
        if ((size > CARD_MIN_AREA) and
            (hierarchy[0][i][3] == -1) and (len(approx) == 4)):
            cardlist[i] = 1
    return contours, cardlist


def cutout(image, pts, width, height):

    #ukladanie rogow prostokata opinajacego karte w odpowiedniej kolejnosci
    rect = np.zeros((4, 2), dtype = "float32")
    w = 200
    h = 300
    s = np.sum(pts, axis = 2)
    
    #sortowanie wierzcholkow (tl - top left itd.)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    #wykrycie czy karta jest poziomo czy pionowo
    if width < height:
        rect[0] = tl
        rect[1] = tr
        rect[2] = br
        rect[3] = bl

    if width > height:
        rect[0] = bl
        rect[1] = tl
        rect[2] = tr
        rect[3] = br

    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0, h-1]], np.float32)
    M = cv2.getPerspectiveTransform(rect,dst)
    warp = cv2.warpPerspective(image, M, (w, h))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('kck',warp)
    return warp


def resize_rs(rs, w, h):
    rs_cnts, hier = cv2.findContours(rs, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rs_cnts = sorted(rs_cnts, key=cv2.contourArea,reverse=True)
    if len(rs_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(rs_cnts[0])
        rs_roi = rs[y1:y1+h1, x1:x1+w1]
        rs_sized = cv2.resize(rs_roi, (h,w), 0, 0)
        return rs_sized
    else:
        print('NOT FOUND')
        return 0







if __name__ == '__main__':
    tranks = []
    tsuits = []
    i=0
    best_diff = 100000
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
                 'Eight','Nine','Ten','Jack','Queen','King']:
        tranks.append(Sampleranks())
        tranks[i].name = Rank
        file = str(i+1) + '.jpg'
        tranks[i].img = cv2.imread('ranks/'+file, cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('kck',tranks[i].img)
        i = i + 1
    i=0
    for Suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']:
        tsuits.append(Samplesuits())
        tsuits[i].name = Suit
        file = 's' + str(i+1) + '.jpg'
        tsuits[i].img = cv2.imread('ranks/'+file, cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('kck',tsuits[i].img)
        i = i + 1
    





    
    im = cv2.imread('cards7.jpg')
    imthresh = pproc(im)
    contours, cardlist = finder(imthresh)
    

    cards = []
    print(cardlist)
    for i in range(len(contours)):
        if (cardlist[i] == 1):
            cards.append(contours[i])

    for i in range(len(cards)):
        cnt = cards[i]
        per = cv2.arcLength(cnt,True)
        epsilon = per/100
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        pts = np.float32(approx)
        x,y,width,height = cv2.boundingRect(cnt)
        avg = np.sum(pts, axis=0)/4
        centr = [int(avg[0][0]),int(avg[0][1])]
        print(centr)
        cimage = cutout(im, pts, width, height)
        #if i==1: cv2.imshow('kck', cimage)


        corner = cimage[0:84,0:24]
        corner = cv2.resize(corner, (0,0), fx=4, fy=4)
        white_level = corner[15,int((24*4)/2)]
        thresh_level = white_level - 30
        if (thresh_level <= 0):
            thresh_level = 1
        retval, cornthresh = cv2.threshold(corner, thresh_level, 255,cv2. THRESH_BINARY_INV)
        #if i==2: cv2.imshow('kck', cornthresh)
        rank = cornthresh[0:180,0:96]
        suit = cornthresh[181:336, 0:96]
        rank = resize_rs(rank,125,70)
        suit = resize_rs(suit,100,70)
        #cv2.imshow('kck',im)
        
        for item in tranks:
            diff = cv2.absdiff(rank, item.img)
            rank_diff = int(np.sum(diff)/255)
            if rank_diff < best_diff:
                best_diff = rank_diff
                best_name = item.name
        print(best_name)
        best_diff = 10000
        for item in tsuits:
            diff = cv2.absdiff(suit, item.img)
            rank_diff = int(np.sum(diff)/255)
            if rank_diff < best_diff:
                best_diff = rank_diff
                best_suit = item.name
        print(best_suit)
        best_diff = 10000

        cv2.putText(im, (best_name + ' of '),
                    (centr[0]-80, centr[1]-10),5,1,(255,0,0))
        cv2.putText(im, (best_suit),
                    (centr[0]-30, centr[1]+10),5,1,(255,0,0))
    cv2.drawContours(im,cards, -1, (255,0,0), 2)
    cv2.imshow('kck', im)
