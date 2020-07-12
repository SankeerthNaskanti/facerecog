 base_model.summary()
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________

summary() function shoes the summary of the model

i have made all the layers trainable to false so that we dont need to train the model again from the scratch

for (i,layer) in enumerate(base_model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)

0 InputLayer False
1 Conv2D False
2 Conv2D False
3 MaxPooling2D False
4 Conv2D False
5 Conv2D False
6 MaxPooling2D False
7 Conv2D False
8 Conv2D False
9 Conv2D False
10 MaxPooling2D False
11 Conv2D False
12 Conv2D False
13 Conv2D False
14 MaxPooling2D False
15 Conv2D False
16 Conv2D False
17 Conv2D False
18 MaxPooling2D False
we can add our own layers to the above architecture

from keras.layers import Dense, Flatten
from keras.models import Sequential


top_model = base_model.output
top_model = Flatten()(top_model)
top_model = Dense(512, activation='relu')(top_model)   
top_model = Dense(256, activation='relu')(top_model)   
top_model = Dense(128, activation='relu')(top_model) 
top_model = Dense(2, activation='softmax')(top_model)  
top_model


from keras.models import Model
model = Model(inputs=base_model.input, outputs=top_model)
model.summary()
if we see the out put of the new model we can see the layers added by us

Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               12845568  
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_3 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 258       
=================================================================
Total params: 27,724,738
Trainable params: 13,010,050
Non-trainable params: 14,714,688
_________________________________________________________________




from keras.preprocessing.image import ImageDataGenerator
The ImageDataGenerator class is very useful in image classification. There are several ways to use this generator, depending on the method we use, here we will focus on flow_from_directory takes a path to the directory containing images sorted in sub directories and image augmentation parameters.

train_datagen = ImageDataGenerator(
    
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1./255)



train_data="C:/Users/DELL-PC/Desktop/mlops-ws/trainingimages"
train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(224, 224),
       
        class_mode='categorical')
Found 400 images belonging to 2 classes.
similarly for validation test

test_datagen = ImageDataGenerator(rescale=1./255)
	test_data="C:/Users/DELL-PC/Desktop/mlops-ws/testingimages"
	test_generator = test_datagen.flow_from_directory(
	        test_data,
	        target_size=(224, 224),
	        class_mode='categorical',
	        shuffle=False)
Found 53 images belonging to 2 classes.



from keras.optimizers import RMSprop
	model.compile(optimizer = RMSprop(lr=0.0001),
	                 loss = 'categorical_crossentropy',
	                 metrics =['accuracy']

	                )
model.fit_generator(train_generator, epochs=2,steps_per_epoch=20, validation_data=test_generator,
	                                validation_steps = 53)


from keras.models import load_model
	from keras.preprocessing import image
	from keras.preprocessing.image import load_img, img_to_array
	                                                                               from numpy import array, expand_dims
testing the image

import cv2
	model = load_model("vggsafe.h5")
	testing_image = "C:/Users/DELL-PC/Desktop/mlops-ws/testingimages/lucky/image0.jpg"
	image = load_img(testing_image, target_size=(224, 224))
	image.show(testing_image)
	

	image = array(image)
	image = expand_dims(image, axis=0)
	if model.predict(image)[0][0] > 0.9:
	    print("not wearing helmet")
	if model.predict(image)[0][1] > 0.9:
	    print("wearing helmet")
	



not wearing helmet
