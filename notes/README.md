# Notes on Developing a Machine Learning Project

This markdown contains notes and insights on developing a machine learning project. It covers the key steps and considerations involved in the process, from data collection and preprocessing to model
selection and evaluation.

## Dataset

[Link here!](https://drive.google.com/drive/folders/1hL8ivxUm_Jvi7xv0d7JG4t47h623T58q?usp=share_link)

## Split Dataset

available at [`data_splitting.ipynb`](../scripts/data_splitting.ipynb)

the output is 3 folders = `train, test, and valid`

## Dataset Creation

### Load Image

```python
def load_image(image_path: str) -> tf.Tensor:

    '''
    The task of the function is to load the image present in the specified given image path. Loading the image the function also performed some
    preprocessing steps such as resizing and normalization.

    Argument:
        image_path(str) : This is a string which represents the location of the image file to be loaded.

    Returns:
        image(tf.Tensor) : This is the image which is loaded from the given image part in the form of a tensor.
    '''

    # Check if image path exists
    assert os.path.exists(image_path), f'Invalid image path: {image_path}'

    # Read the image file
    image = tf.io.read_file(image_path)

    # Load the image
    try:
        image = tfi.decode_jpeg(image, channels=3)
    except:
        image = tfi.decode_png(image, channels=3)

    # Change the image data type
    image = tfi.convert_image_dtype(image, tf.float32)

    # Resize the Image
    image = tfi.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Convert image data type to tf.float32
    image = tf.cast(image, tf.float32)

    return image
```

### Load Dataset

```python
def load_dataset(root_path: str, class_names: list, trim: int=None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load and preprocess images from the given root path and return them as numpy arrays.

    Args:
        root_path (str): Path to the root directory where all the subdirectories (class names) are present.
        class_names (list): List of the names of all the subdirectories (class names).
        trim (int): An integer value used to reduce the size of the data set if required.

    Returns:
        Two numpy arrays, one containing the images and the other containing their respective labels.
    '''

    if trim:
        # Trim the size of the data
        n_samples = len(class_names) * trim
    else:
        # Collect total number of data samples
        n_samples = sum([len(os.listdir(os.path.join(root_path, name))) for name in class_names])

    # Create arrays to store images and labels
    images = np.empty(shape=(n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    labels = np.empty(shape=(n_samples, 1), dtype=np.int32)

    # Loop over all the image file paths, load and store the images with respective labels
    n_image = 0
    for class_name in tqdm(class_names, desc="Loading"):
        class_path = os.path.join(root_path, class_name)
        image_paths = list(glob(os.path.join(class_path, "*")))[:trim]
        for file_path in image_paths:
            # Load the image
            image = load_image(file_path)

            # Assign label
            label = class_names.index(class_name)

            # Store the image and the respective label
            images[n_image] = image
            labels[n_image] = label

            # Increment the number of images processed
            n_image += 1

    # Shuffle the data
    indices = np.random.permutation(n_samples)
    images = images[indices]
    labels = labels[indices]


    return images, labels
```

### Use the function (load dataset)

```python
# Load the training dataset
X_train, y_train = load_dataset(root_path = train_dir, class_names = class_names)

# # Load the validation dataset
X_valid, y_valid = load_dataset(root_path = valid_dir, class_names = class_names)

# Load the testing dataset
X_test, y_test = load_dataset(root_path = test_dir, class_names = class_names)
```

## Testing on Model

```python
test_loss, test_acc = xception.evaluate(X_test, y_test)
print("Loss    : {:.4}".format(test_loss))
print("Accuracy: {:.4}%".format(test_acc*100))
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
best_xception = tf.keras.models.load_model('model.h5')
best_xception.summary()
```
