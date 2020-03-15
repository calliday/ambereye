f = open("units_new.txt", "w")
temp = {}
f.write("image_name, class, color,\n")
for row in range(10):
    f.write("{}, {}, {},\n".format(row+10, row+20, row+30))
f.close()
