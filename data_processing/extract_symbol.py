import pickle
import os
from process_images import ImageProcessing
from rules import Rules

'''
generete train and test pickle files for train model and test model
'''

rules = Rules().get_rules()
train_file = os.getcwd() + "/data/train/"
test_file = os.getcwd() + "/data/test/"
extra_train_file = os.getcwd() + "/data/symbols"

train_data = {}
test_data = {}

train_symbol = {}
test_symbol = {}

train_results = {}
test_results = {}

def process_train_data():
    for item in os.listdir(train_file):
        if item.endswith(".png"):
            im1, _ = ImageProcessing(train_file).process_image(item) 
            train_results[item] = im1
            train_symbol[item] = rules[item.split(".")[0].split("_")[0]]


    for subdir, _, files in os.walk(extra_train_file):
        for item in files:
            if item.endswith(".png"):
                im1, _ = ImageProcessing(subdir+"/").process_image(item) 
                train_results[item] = im1
                e = subdir.split("/")
                print( e[len(e)-1])
                train_symbol[item] = rules[e[len(e)-1]]

    train_data["images"] = train_results
    train_data["labels"] = train_symbol

    pf = open("train_data.pkl","wb")
    pickle.dump(train_data, pf)

def process_test_data():
    for item in os.listdir(test_file):
        if item.endswith(".png"):
            im1, _ = ImageProcessing(test_file).process_image(item)
            test_results[item] = im1
            test_symbol[item] = rules[item.split(".")[0].split("_")[0]]
    test_data["images"] = test_results
    test_data["labels"] = test_symbol
    
    pf = open("test_data.pkl","wb")
    pickle.dump(test_data, pf)

def main():
    process_train_data()
    process_test_data()

if __name__ == "__main__":
    main()
