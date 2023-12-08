import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydot
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


print("------------------------------------------------------")

img_dir=r"E:\np\IMPORTANT\cropped"
LM10=os.listdir(img_dir+"//lionel_messi")
MARI=os.listdir(img_dir+"//maria_sharapova")
RGFT=os.listdir(img_dir+"//roger_federer")
VK18=os.listdir(img_dir+"//virat_kohli")
SRN=os.listdir(img_dir+"//serena_williams")

print("2------------------------------------------------------")


print("Then lenght of Lionel Messi is",len(LM10))
print("Then lenght of Maria is",len(MARI))
print("Then lenght of Roger Federer is",len(RGFT))
print("Then lenght of Virat Kohli is",len(VK18))
print("Then lenght of Serena Williams is",len(SRN))

print("3------------------------------------------------------")

dataset = []
label = []
img_size=(224,224)


for i,image_name in tqdm(enumerate(LM10),desc="Lional Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(0)

for i,image_name in tqdm(enumerate(MARI),desc="Maria Sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(1)

for i,image_name in tqdm(enumerate(RGFT),desc="Roger Federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(2)

for i,image_name in tqdm(enumerate(VK18),desc="Virat Kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(3)

for i,image_name in tqdm(enumerate(SRN),desc="Serena Williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(img_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(4)

dataset=np.array(dataset)
label=np.array(label)


print("---------------------------------------")

print("lenght of the dataset is ",len(dataset))
print("lenght of the dataset is",len(label))

print("---------------------------------------------")


x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)


x_train=x_train.astype('float')/255
x_test=x_test.astype('float')/255 

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32,kernel_size =(3,3),activation="ReLU",input_shape=(224,224,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters=64,kernel_size =(3,3),activation="ReLU"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters=128,kernel_size =(3,3),activation="ReLU"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters=128,kernel_size =(3,3),activation="ReLU"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512,activation="ReLU"))          
model.add(keras.layers.Dense(5,activation="softmax"))


model.summary()

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


early_stooping_cb=keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history=model.fit(x_train,y_train,epochs=30,batch_size =64,validation_split=0.1,callbacks=[early_stooping_cb])

print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)


def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((224,224))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction,axis=1)[0]
    class_name = ['lionel_messi','maria_sharapova','roger_federer','virat_kohli','serena_williams']
    predicted_class_name = class_name[predicted_class]
    return predicted_class_name

    
# Make predictions on new images
image_paths = [
    r'E:\np\IMPORTANT\cropped\maria_sharapova\maria_sharapova1.png',
    r'E:\np\IMPORTANT\cropped\lionel_messi\lionel_messi1.png',
    r'E:\np\IMPORTANT\cropped\roger_federer\roger_federer1.png',
    r'E:\np\IMPORTANT\cropped\serena_williams\serena_williams6.png',
    r'E:\np\IMPORTANT\cropped\virat_kohli\virat_kohli10.png'
]

for image_path in image_paths:
    image_filename = os.path.basename(image_path)  # Extract filename from image path
    predicted_class_label = make_prediction(image_path, model)
    print(f"Predicted label for image {image_filename}: {predicted_class_label}")


