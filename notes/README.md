# Notes on Developing a Machine Learning Project

This markdown contains notes and insights on developing a machine learning project. It covers the key steps and considerations involved in the process, from data collection and preprocessing to model
selection and evaluation.

## Dataset

`https://drive.google.com/drive/folders/1hL8ivxUm_Jvi7xv0d7JG4t47h623T58q?usp=share_link`

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

## Avoid error on Image Byte Loss

```python
# avoid error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

## Testing on Model

```python
# specify path
test_data_dir = 'test/test/'

# create ImageDataGenerator
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=32,
        class_mode='categorical')

# evaluate
xtest_loss, xtest_acc = model.evaluate(test_generator)
print(f"Xception Baseline Testing Loss     : {xtest_loss}.")
print(f"Xception Baseline Testing Accuracy : {xtest_acc}.")
```

## Apply Hypertuning

### Build Model Function

```python
def build_model(hp, n_classes=13):

    # Define all hyperparms
    n_layers = hp.Choice('n_layers', [0, 2, 4])
    dropout_rate = hp.Choice('rate', [0.2, 0.4, 0.5, 0.7])
    n_units = hp.Choice('units', [64, 128, 256, 512])

    # Mode architecture
    model = Sequential([
        xception,
        GlobalAveragePooling2D(),
    ])

    # Add hidden/top layers
    for _ in range(n_layers):
        model.add(Dense(n_units, activation='relu', kernel_initializer='he_normal'))

    # Add Dropout Layer
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(n_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = Adam(LEARNING_RATE),
        metrics = ['accuracy']
    )

    # Return model
    return model
```

### Apply Random Searcher

```python
# Initialize Random Searcher
random_searcher = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=10,
    seed=42,
    project_name="XceptionSearch",
    # loss='sparse_categorical_crossentropy'
    loss='categorical_crossentropy'
)

# Start Searching
search = random_searcher.search(
    train_generator,
    validation_data=validation_generator,
    epochs = 10,
    batch_size = BATCH_SIZE
)
```

### Callbacks that need to be imported

```python
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
```

### Take the best model from the random search

```python
# Collect the best model Xception Model Architecture obtained by Random Searcher
best_xception = build_model(random_searcher.get_best_hyperparameters(num_trials=1)[0])

# Model Architecture
best_xception.summary()

# Compile Model
best_xception.compile(
    # loss='sparse_categorical_crossentropy',
    loss='categorical_crossentropy',
    optimizer=Adam(LEARNING_RATE*0.1),
    metrics=['accuracy']
)

# Model Training
best_xception_history = best_xception.fit(
    train_generator,
    validation_data=validation_generator,
    epochs = 50,
    batch_size = BATCH_SIZE*2,
    callbacks = [
        EarlyStopping(patience=2, restore_best_weights=True),
        ModelCheckpoint("BestXception.h5", save_best_only=True)
    ]
)

loss, accuracy = best_xception.evaluate(test_generator)
print(f"Test Loss after Tunig     : {loss}")
print(f"Test Accuracy after Tunig : {accuracy}")
```

### Load the model

```python
#  Load model
best_xception = tf.keras.models.load_model('/BestXception.h5', compile=False)
best_xception.summary()
```
