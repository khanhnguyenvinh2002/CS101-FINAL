items = []
with open("labels.txt") as file:
    for line in file:
        items.append(line.rstrip())
num_failed = 0
total = 0
with open("predictions.txt") as file2:
    for line2 in file2:
        total+=1
        if not (line2.rstrip() in items):
            num_failed+=1
            print(line2.rstrip())

print ("Accuracy is ", ((total-num_failed)/float(total)))
print ("Number of wrong categorization is ", (num_failed))