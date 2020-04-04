import csv
f = open('car_types_classes.py', 'w')
f.write('car_types = {\n')
with open('classes.csv') as handle:
    reader = csv.DictReader(handle)

    for row in reader:
        f.write('\t\'{}\': \'{}\',\n'.format(row['old'], row['new']))

f.write('}\n\n')
f.write('possible_types = [\'SUV\', \'Sedan\', \'Coupe\', \'Truck\', \'Van\', \'Convertible\', \'Hatchback\', \'Wagon\']')
f.close()
