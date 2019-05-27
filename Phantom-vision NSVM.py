
#center_platforme=[240, 312]
#image_range=[180,180]
center_platforme=[380,260]  # Offline params
image_range=[180,180]        # Offline params
##inicializacije
ref=[0.10,0.05]
faktor=0.0005 ## m/pixel
obmocjeTrajektorije=0.6
minR= 8
maxR= 20
maxradij=170 #Max odmik krogle od središča platforme, da bomo še upoštevali rezultate

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
import socket
from codecs import encode, decode
import colorsys
import struct
from tkinter import *
from PIL import Image, ImageTk
from threading import Thread
from tkinter import simpledialog

s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.connect(('192.168.65.97',25000))

def dist(p1,p2):
    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return distance

def traj(image,steprad,krad,pointnum,dth_amp):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),cv.BORDER_CONSTANT)
    thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,7,4)
    points=trackEngine(thresh,steprad,krad,pointnum,dth_amp)
    return(points)

def FixAngle(ang): #Kote, zamaknjene za poljubno period, prestavi med 0 in 2pi
    if ang<0:
        ang=ang%(2*np.pi)
    if ang>2*np.pi:
        ang=ang%(2*np.pi)
    return(ang)
def GroupUp(iList): ##Glede na Poz in Neg fronto združi posamezne skupine svetlih slikovnih elementov
    prev=0
    groupNum=-1
    gstarts=[]
    gstops=[]
    temp=[]
    for ii in range(iList.shape[0]):
        if (prev==0)and(iList[ii]>0):  #POZITIVNA FRONTA
            groupNum+=1
            gstarts.append(ii)
        elif (prev>0)and(iList[ii]==0): #NEGATIVNA FRONTA
            gstops.append(ii-1)
        prev=iList[ii]
    if len(gstarts)!=len(gstops):
        gstops.append(ii)
    for tt in range(len(gstarts)): #Sestavimo seznam začetnih in končnih indeksov svetlega območja
        temp.append((gstarts[tt],gstops[tt]))
    return(temp)


def useKernel(img,krad,runI,runJ,theta_prev):
    KernelW=2*krad+1
    fram=np.copy(img[runJ-krad:runJ+krad+1,runI-krad:runI+krad+1])
    try:
        top=fram[0,:]
    except:
        print("ERR: Trajectory too close to the edge.")
    right=fram[:,-1]
    left=fram[:,0]
    bottom=fram[-1,:]
    groups_top=[]
    groups_bottom=[]
    groups_left=[]
    groups_right=[]
    try:
        groups_top=np.array(GroupUp(top))[::-1,::-1] #Vrne seznam začetnih in končnih indeksov => pretvori v (X,Y)    
    except:
        pass
    try:
        groups_right=np.array(GroupUp(right))[::-1,::-1]
    except:
        pass
    try:
        groups_left=np.array(GroupUp(left))
    except:
        pass
    try:
        groups_bottom=np.array(GroupUp(bottom))
    except:
        pass
    #Skupine robnih indeksov združimo v skupine z [X,Y] koordinatami glede na izhodiščni (centralni) pixel
    gps=[]
    kk=0
    for pair in groups_left:
        gps.append([[0,pair[0]],[0,pair[1]]])
    for pair in groups_bottom:
        if kk==0: 
            if bottom[0]>0:
                gps[-1][1]=[pair[1],KernelW-1]
            else:
                gps.append([[pair[0],KernelW-1],[pair[1],KernelW-1]])
            kk=1
        else:
            gps.append([[pair[0],KernelW-1],[pair[1],KernelW-1]])
    kk=0
    for pair in groups_right:
        if kk==0:
            if right[-1]>0:
                gps[-1][1]=[KernelW-1,pair[1]]
            else:
                gps.append([[KernelW-1,pair[0]],[KernelW-1,pair[1]]])
            kk=1
        else:
            gps.append([[KernelW-1,pair[0]],[KernelW-1,pair[1]]])
    kk=0
    for pair in groups_top:
        if kk==0:
            if top[-1]>0:
                gps[-1][1]=[pair[1],0]
            else:
                gps.append([[pair[1],0],[pair[1],0]])
            kk=1
        else:
            gps.append([[pair[1],0],[pair[1],0]])
    if top[0]>0:
        #Stitch at (0,0) -- left top
        gps[0][0]=gps[-1][1]
        del gps[-1]
    
    #Skupine koordinat pretvorimo v skupine kotov
    coords=np.array(gps)-krad
    angles=[]
    avgangs=[]
    for cds in coords:
        th1=np.arctan2(-cds[0][1],cds[0][0])
        th2=np.arctan2(-cds[1][1],cds[1][0])
        angles.append([th1,th2])
        avgangs.append((th1+th2)/2)
        if (angles[-1][1]-angles[-1][0])<0:
            if avgangs[-1]>0:
                avgangs[-1]=avgangs[-1]-np.pi
            else:
                avgangs[-1]=avgangs[-1]+np.pi
    angdiffs=[]
    for ang in avgangs:
        angdiffs.append(FixAngle(ang-theta_prev)) #razpon od 0 do 2pi
        if angdiffs[-1]>np.pi:
            angdiffs[-1]-=2*np.pi
    angdiffs_abs=np.abs(np.array(angdiffs))
    angle_index=0
    try:
        angle_index=np.argmin(angdiffs_abs)
    except:
        print("ERR: No line found in kernel!")
        return(0,0)
    return(avgangs[angle_index],angdiffs[angle_index],len(gps))

def trackEngine(img,steprad,krad,pointnum,dth_amp):
    startI=-1
    startJ=-1
    for i in range(2,img.shape[0]):  ##Iskanje 1. točke (počasi se spuščamo po sliki, iščemo prvi svetel piksel)
        if startI>=0:
            break
        for j in range(0,img.shape[1]):
            if startJ>=0:
                break
            if img[i,j]>0:
                startI=j
                startJ=i
    run=True
    runJ,runI=startJ,startI
    theta_old=0
    dtheta_old=0
    rr=0
    points=[]
    theta_start=0
    closed_loop1=False
    closed_loop2=False
    #Assemble a list of chain points (trajectory)
    while (run):
        theta_old=FixAngle(theta_old+(dtheta_old*dth_amp))
        theta_new,dtheta_new,num_edges=useKernel(img,krad,runI,runJ,theta_old)
        if rr==0:
            theta_start=theta_new
        rr+=1
        runI=int(round(runI+steprad*np.cos(theta_new)))
        runJ=int(round(runJ-steprad*np.sin(theta_new)))
        points.append([runI,runJ]) # [ [x1,y1], [x2,y2], ... ]
        dtheta_old=dtheta_new
        theta_old=theta_new
        if (rr>4)and(dist([runI,runJ],points[0])<=1.5*krad):        #Loop closed!
            run=False
            closed_loop1=True
        if (num_edges<2):                                           #End of the line (exits kernel only on 1 side)
            run=False
        if(rr>=600):                                                #Force stop after 600 iterations -> check for overlaps (to avoid endless loop)
            run=False
            points=OverlapCheck(points,steprad,pointnum)
    run=True
    pointsReverse=[]
    runI,runJ=points[0][0],points[0][1]
    theta_old=FixAngle(theta_start+np.pi)
    rr=0
    while (run):
        theta_old=FixAngle(theta_old+(dtheta_old*dth_amp))
        theta_new,dtheta_new,num_edges=useKernel(img,krad,runI,runJ,theta_old)
        runI=int(runI+steprad*np.cos(theta_new))
        runJ=int(runJ-steprad*np.sin(theta_new))
        pointsReverse.append([runI,runJ])
        rr+=1
        dtheta_old=dtheta_new
        theta_old=theta_new
        if (rr>4)and(dist([runI,runJ],points[0])<=1.5*krad):        #Loop closed!
            run=False
            closed_loop2=True
        if (num_edges<2)and(rr>0):                                  # Found ending
            run=False
        if(rr>=600):
            run=False
            pointsReverse=OverlapCheck(pointsReverse,steprad,pointnum)
    
    if closed_loop1 and closed_loop2: #Če z obeh strani pridemo do začetka, smo naredili 2 zanki - upoštevamo le eno
        oPoints=points
    else:
        oPoints=pointsReverse[::-1]+points
    return(oPoints)


def pltTraj(img,trajec): #Izriše barvne točke na sliko, s spreminjanjem Hue komponente HSV spektra
    aa=0
    bgr=[]
    pntlast=[]
    for pnt in trajec:
        bgr=[]
        rgb=[]
        rgb = colorsys.hsv_to_rgb(aa/50.0, 1, 1)
        #cv.circle(img,(pnt[0],pnt[1]),0,(rgb[0]*255,rgb[1]*255,rgb[2]*255),5)  #Izrišemo krogce na sliko
        if aa>0: #Trajektorijo povežemo s črtami
            cv.line(img, (pntlast[0],pntlast[1]), (pnt[0],pnt[1]), (255-rgb[0]*255,255-rgb[1]*255,255-rgb[2]*255),2)
        #img[pnt[1],pnt[0]]=[rgb[0]*255,rgb[1]*255,rgb[2]*255]  # Pobarvamo pixle na trajektoriji
        pntlast=pnt
        aa+=1
    return img

def OverlapCheck(trajectory,steprad,pointnum): #parametri: array točk trajektorije, kernel radij, št. bližnjih točk za izločitev
    count=0
    ii=0
    overlap=False
    for ii in range(pointnum,len(trajectory)):  ##Za vsako točko v trajectory
        blizu=False
        for jj in range(len(trajectory[:ii-pointnum])):       ##izračunamo razdaljo do vseh predhodnih točk
            if (dist(trajectory[ii],trajectory[jj])<=steprad/2):
                blizu=True
                break
        if blizu:
            count+=1
            if count>=pointnum: #Če ima več zaporednih točk sosede v bližnji okolici, zaznamo prekrivanje
                overlap=True
                break
        else:
            count=0
    if overlap:
        trajectory=trajectory[:ii-pointnum+2]
        print(f"OVERLAP found:   {ii} ")
    return(trajectory)



########################################################################
########################################################################
######################                       ###########################
######################  GLAVNI DEL PROGRAMA  ###########################
######################                       ###########################
########################################################################
########################################################################

########################################################################
######################         GUI           ###########################
########################################################################

root = Tk()

def platform(): ####kalibracija platforme
    y = 180
    x = 180
    global center_platforme, image_range,faktor
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(3,3),cv.BORDER_DEFAULT)
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,20,param1=80,param2=10,minRadius=150,maxRadius=250)
    plt.figure(0)
    cv.circle(frame,(circles[0,0,0],circles[0,0,1]),circles[0,0,2],(0,255,0),1)
    faktor=0.15/circles[0,0,2]
    print(f"Faktor: {faktor} m/px")
    cv.circle(frame,(circles[0,0,0],circles[0,0,1]),2,(0,0,255),3)
    center_platforme=[int(circles[0,0,1]),int(circles[0,0,0])]
    image_range=[int(circles[0,0,2]),int(circles[0,0,2])]  #Izrez slike +-radij platforme
    cv.imshow('platforma',frame)
    var1.set("✓")
    var11.set(f"{center_platforme}")

def ball(): ###kalibracija žogice
    global minR,maxR
    ret, frame = cam.read()
    frame=np.copy(frame[center_platforme[0]-image_range[0]:center_platforme[0]+image_range[0]+1,center_platforme[1]-image_range[1]:center_platforme[1]+image_range[1]+1])
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(3,3),cv.BORDER_DEFAULT)
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,20,param1=60,param2=10,minRadius=5,maxRadius=30)
    radij=int(circles[0,0,2])
    minR= radij-3
    maxR= radij+5
    plt.figure(0)
    cv.circle(frame,(circles[0,0,0],circles[0,0,1]),circles[0,0,2],(255,0,0),1)
    cv.circle(frame,(circles[0,0,0],circles[0,0,1]),minR,(0,0,255),1)
    cv.circle(frame,(circles[0,0,0],circles[0,0,1]),maxR,(0,0,255),1)
    cv.imshow('platforma',frame)
    var2.set("✓")
    var21.set(f"{radij}")

trajektorijaC = [] 
trajektorijaIMG = []

def trajectory(): ###zaznava trajektorije
    global trajektorijaC, trajektorijaIMG
    ret, frame = cam.read()
    img=np.copy(frame[center_platforme[0]-int(image_range[0]*obmocjeTrajektorije):center_platforme[0]+int(image_range[0]*obmocjeTrajektorije)+1,center_platforme[1]-int(image_range[1]*obmocjeTrajektorije):center_platforme[1]+int(image_range[1]*obmocjeTrajektorije)+1])
    kernel_rad= simpledialog.askinteger("Input number", "Enter kernel radius (9)")
    step_rad=kernel_rad #simpledialog.askinteger("Input number", "Enter interpolation radius (9)")
    overlap_thresh=simpledialog.askinteger("Input number", "Min number of consecutive near points (5)")
    print(img.shape)
    trajectory=traj(img,step_rad,kernel_rad,overlap_thresh,0.0) ##Robustno pri kernel radius = 6
    print(f"Skupna dolžina trajektorije: {len(trajectory)} ")
    #trajectory=OverlapCheck(trajectory,kernel_rad,overlap_thresh)
    img=pltTraj(img,trajectory)
    plt.figure(5)
    cv.imshow("slika trajektorije",img)
    trajektorijaC   = []
    trajektorijaIMG = [] 
    for tocka in trajectory:
        trajektorijaC.append([tocka[0]-int(image_range[1]*obmocjeTrajektorije),tocka[1]-int(image_range[0]*obmocjeTrajektorije)])
    for tocka in trajektorijaC:
        trajektorijaIMG.append([tocka[0]+image_range[1],tocka[1]+image_range[0]])
    var3.set("✓")

cam = cv.VideoCapture(0)    



def run():
    global a
    a = True
    
    #Možnost shranjevanja videa v datoteko:
    #fourcc = cv.VideoWriter_fourcc(*'XVID')
    #out = cv.VideoWriter('outputFast.avi', fourcc, 20.0, (2*image_range[1],2*image_range[0]))
    
    var4.set("Running")
    var5.set("")
    last_time = time.time()
    cent_prev=[0,0]
    der = [0,0]
    start_time = time.time()
    vodi_index=0
    vodi_dt=0.2
    trun=0
    
    while(a):
        ret, frame = cam.read()
        frame=frame[center_platforme[0]-image_range[0]:center_platforme[0]+image_range[0]+1,center_platforme[1]-image_range[1]:center_platforme[1]+image_range[1]+1]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
        circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,2,20,param1=54,param2=10,minRadius=minR,maxRadius=maxR)## Iskanje žogice
        frame=pltTraj(frame,trajektorijaIMG)
        tim = time.time()
        trun=tim-start_time
        vodi_index=int(math.floor(trun/vodi_dt))
        if vodi_index<len(trajektorijaC):
            ref=[trajektorijaC[vodi_index][0]*faktor,trajektorijaC[vodi_index][1]*faktor]
        else:
            ref=[0,0]
        cv.circle(frame,(trajektorijaIMG[vodi_index][0],trajektorijaIMG[vodi_index][1]),2,(0,0,255),4)
        try:
            cv.circle(frame,(circles[0,0,0],circles[0,0,1]),minR,(0,0,255),1)
            cv.circle(frame,(circles[0,0,0],circles[0,0,1]),circles[0,0,2],(255,0,0),1)
            cv.circle(frame,(circles[0,0,0],circles[0,0,1]),maxR,(0,0,255),1)
            cv.circle(frame,(image_range[0],image_range[1]),maxradij,(255,0,255),4)
            der[0]=(circles[0,0,0]-cent_prev[0])/(tim-last_time)
            der[1]=(circles[0,0,1]-cent_prev[1])/(tim-last_time)
            cent_prev=circles[0,0]
            if dist((circles[0,0,0],circles[0,0,1]),(image_range[0],image_range[1]))<maxradij:
                cv.line(frame, (trajektorijaIMG[vodi_index][0],trajektorijaIMG[vodi_index][1]), (circles[0,0,0],circles[0,0,1]), (0,255,0),3)
                vals = np.array([ref[0], ref[1], (circles[0,0,0]-image_range[0])-trajektorijaC[vodi_index][0], (circles[0,0,1]-image_range[1])-trajektorijaC[vodi_index][1],der[0],der[1]])
                vals = vals.reshape(6,1)
                bin_vals = struct.pack('<6f',*vals)
                s.send(bin_vals)
                var6.set(f"{vals}")
                
            else:
                vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                vals = vals.reshape(6,1)
                bin_vals = struct.pack('<6f',*vals)
                s.send(bin_vals)
            cv.circle(frame,(circles[0,0,0],circles[0,0,1]),circles[0,0,2],(100,0,0),1)
        except:
            print("Ball not detected")
        try:
            print("Frequency = {0:.1f}, dt= {0:.1f}".format(1.0/(tim-last_time),(tim-last_time)))
        except:
            pass
        last_time=tim
        plt.figure(4)
        cv.imshow('frame',frame)
        #out.write(frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            #out.release()
            cv.destroyAllWindows()
            s.close
            break

def stop():
    global a
    global b
    a = False
    b = False
    var4.set("")
    var5.set("")
    var7.set("Stopped")

root.title("Robotski Vid / Vodenje robotov")
var1 = StringVar()
var2 = StringVar()
var11 = StringVar()
var21= StringVar()
var3 = StringVar()
var4 = StringVar()
var5 = StringVar()
var6 = StringVar()
var7 = StringVar()

label0 = Label(root, text = "Pozicioniranje objekta z robotom na podlagi vizualne povratne zanke").grid(row = 0, columnspan = 4)

Button1 = Button(root, text = "Kalibracija platforme", width = 25, command = platform).grid(row = 1, column = 0, columnspan = 2)
label1 = Label(root, textvariable = var1, fg = "green").grid(row = 1, column = 2)
label11 = Label(root, textvariable = var11).grid(row = 1, column = 3)

Button2 = Button(root, text = "Kalibracija krogle", width = 25, command = ball).grid(row = 2, column = 0, columnspan = 2)
label2 = Label(root, textvariable = var2, fg = "green").grid(row = 2, column = 2)
label21 = Label(root, textvariable = var21).grid(row = 2, column = 3)

Button3 = Button(root, text = "Zaznava trajektorije", width = 25, command = trajectory).grid(row = 3, column = 0, columnspan = 2)
label3 = Label(root, textvariable = var3, fg = "green").grid(row = 3, column = 2)

Button4 = Button(root, text = "Vodi po trajektoriji", width = 20, bg = "green", command = lambda: Thread(target = run).start()).grid(row = 4, column = 0, columnspan = 2)

Button5 = Button(root, text = "Stop", width = 15, bg = "red", command = stop).grid(row = 4, column = 4)

label4 = Label(root, textvariable = var4).grid(row = 5, column = 0, columnspan = 2)
label5 = Label(root, textvariable = var5).grid(row = 5, column = 2, columnspan = 2)
label5 = Label(root, textvariable = var7).grid(row = 5, column = 4)

label61 = Label(root, text = "Poslane spr.").grid(row = 0, column = 4)
label6 = Label(root, textvariable = var6).grid(row = 1, column = 4, rowspan = 3)

label7 = Label(root, text = "Ob kliku na gumb Stop je potreben ponoven zagon programa in ponovna kalibracija").grid(row = 6, column = 0, columnspan = 5)

root.mainloop()
