# Notes on Developing a Machine Learning Project

This markdown contains notes and insights on developing a machine learning project. It covers the key steps and considerations involved in the process, from data collection and preprocessing to model 
selection and evaluation.

## Dataset

`https://drive.google.com/file/d/1UOzAAQcEjztiVjO2DA4ms0bttrG4Z0ot/view?usp=share_link`

## Split Dataset

```python
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs)
```
