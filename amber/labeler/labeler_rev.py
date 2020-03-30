import tkinter as tk
from PIL import Image
from PIL import ImageTk
from TargetManagerRev import Manager
import os

os.chdir("/home/pi/ambereye/amber/labeler")

win = tk.Tk()
win.title("Select a color")

def getNextImg(carImg=None):
    carImg = Image.open('car_ims/' + managa.getNextImgName())
    s= carImg.size
    ratio = 480/s[1]
    carImg = ImageTk.PhotoImage(carImg.resize((int(s[0]*ratio), int(s[1]*ratio)), Image.ANTIALIAS))
    return carImg

managa = Manager(csvPath='targets_rev.csv')
carImg = None
carImg = getNextImg(carImg)
imgLabel = tk.Label(image=carImg)
imgLabel.grid(row=0, column=1, rowspan=14)


def clicked(color=None, imgLabel=None):
    if color != None:
        print(color)
        managa.saveCar(managa.getNextImgName(), color)
        carImg = None
        carImg = getNextImg(carImg)
        imgLabel.configure(image=carImg)
        imgLabel.image = carImg
    else:
        print("Going Back")
        managa.removeCar()
        carImg = None
        carImg = getNextImg(carImg)
        imgLabel.configure(image=carImg)
        imgLabel.image = carImg

buttons = []
colors = ["White", "Silver",  "Black",   "Gray",    "Blue",    "Red",     "Green",   "Brown",   "Tan",     "Purple",  "Pink",    "Orange", "Yellow"]
hexes  = ["white", "#c0c0c0", "#000000", "#666666", "#7084fa", "#fb6565", "#228b22", "#80460f", "#d2b48c", "#9400d3", "#ff69b4", "#ff9229", "#f6f074"]
fcolor = ["black", "black",   "white",   "white",   "white",   "white",   "white",   "white",   "black",   "white",   "black",   "white",   "black"]

buttons.append(tk.Button(win, text=colors[0], command=lambda: clicked(colors[0], imgLabel), bg=hexes[0], fg=fcolor[0]))
buttons.append(tk.Button(win, text=colors[1], command=lambda: clicked(colors[1], imgLabel), bg=hexes[1], fg=fcolor[1]))
buttons.append(tk.Button(win, text=colors[2], command=lambda: clicked(colors[2], imgLabel), bg=hexes[2], fg=fcolor[2]))
buttons.append(tk.Button(win, text=colors[3], command=lambda: clicked(colors[3], imgLabel), bg=hexes[3], fg=fcolor[3]))
buttons.append(tk.Button(win, text=colors[4], command=lambda: clicked(colors[4], imgLabel), bg=hexes[4], fg=fcolor[4]))
buttons.append(tk.Button(win, text=colors[5], command=lambda: clicked(colors[5], imgLabel), bg=hexes[5], fg=fcolor[5]))
buttons.append(tk.Button(win, text=colors[6], command=lambda: clicked(colors[6], imgLabel), bg=hexes[6], fg=fcolor[6]))
buttons.append(tk.Button(win, text=colors[7], command=lambda: clicked(colors[7], imgLabel), bg=hexes[7], fg=fcolor[7]))
buttons.append(tk.Button(win, text=colors[8], command=lambda: clicked(colors[8], imgLabel), bg=hexes[8], fg=fcolor[8]))
buttons.append(tk.Button(win, text=colors[9], command=lambda: clicked(colors[9], imgLabel), bg=hexes[9], fg=fcolor[9]))
buttons.append(tk.Button(win, text=colors[10], command=lambda: clicked(colors[10], imgLabel), bg=hexes[10], fg=fcolor[10]))
buttons.append(tk.Button(win, text=colors[11], command=lambda: clicked(colors[11], imgLabel), bg=hexes[11], fg=fcolor[11]))
buttons.append(tk.Button(win, text=colors[12], command=lambda: clicked(colors[12], imgLabel), bg=hexes[12], fg=fcolor[12]))

buttons.append(tk.Button(win, text="Oops", command=lambda: clicked(None, imgLabel), bg="black", fg="white"))

for i, button in enumerate(buttons):
   button.grid(row=i, column=0)

win.mainloop()
