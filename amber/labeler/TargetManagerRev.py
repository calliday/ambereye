import csv

class Manager:
    def __init__(self, csvPath='', step=1):
        self.csvPath = csvPath
        self.step = 1
        return
    
    # get the index of the next car to work on
    def getNextImgName(self):
        with open(self.csvPath, 'r') as targets:
            lines = targets.readlines()
            if len(lines) > 1:
                lines.pop
            numba = 16185 - (len(lines) * self.step)
            return (str(numba).zfill(6) + '.jpg')

    # write the car details to the csv
    def saveCar(self, image="", label=""):
        with open(self.csvPath, 'a') as targets:
            targets.write(image + ',' + label + '\n')
        return

    # remove the last car in the csv
    def removeCar(self):
        f = open(self.csvPath, 'r+')
        lines = f.readlines()
        if (lines == []):
            return
        lines.pop()
        f = open(self.csvPath, 'w+')
        f.writelines(lines)
        return

