from skimage.transform import resize
from PIL import Image
import numpy as np

image_paths = []
image_labels = []
root_directory = "./dataset/"

def create_paths(image_data):
    image_num = int(image_data[1])
    image_pre_nums = ''
    if(image_num < 10):
        image_pre_nums = "00"
    if(image_num < 100 and image_num >= 10):
        image_pre_nums = "0"
    
    return root_directory + image_data[0] +"-cam0-rgb/" + image_data[0] + "-cam0-rgb-"+ image_pre_nums + image_data[1] +".png"
def truncate(value):
    if(value < 0):
        return 0
    return value

with open("./"+root_directory+"/features/urfall-cam0-adls.csv") as file: 
    data = file.read()
    filtered = list(filter(lambda x:  x, data.split("\n")))
    mapped = list(filter(lambda x: int(x[2]) != 0,map(lambda x: [x.split(',')[0],x.split(',')[1], x.split(',')[2]], filtered)))
    image_paths = list(map(create_paths, mapped))
    image_labels = list(map(lambda x: int(x[2]), mapped))

images = []
x_train = []
x_test = []
Y_train = []
Y_test = []
counter = 1
for index, x in enumerate(image_paths):
    print(x)
    image_array = np.array(Image.open(x).resize((227,227)))
    if(counter == 4):
        x_test.append(image_array)
        Y_test.append(truncate(image_labels[index]))
        counter=1
    else:
        x_train.append(image_array)
        Y_train.append(truncate(image_labels[index]))
        counter = counter + 1

print(np.shape(x_test))
print(np.shape(image_labels))

with open("fall_dataset", 'wb') as file:
    np.save(file,(x_train, x_test, Y_train, Y_test), allow_pickle=True)
