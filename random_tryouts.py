# Me doing debugging in a seperate file lol

file = "project/validation_images/labels.csv"
import csv
with open(file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    classid_dict = dict()
    for row in spamreader:
        classid_dict.update({row[0]: row[1]})

print(len(classid_dict))
