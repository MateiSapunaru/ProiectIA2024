
```markdown
# Retinal OCT Image Classification (Keras + CNN)

This project implements an AlexNet-style Convolutional Neural Network using Keras to classify retinal OCT images into four categories:

- CNV  
- DME  
- DRUSEN  
- NORMAL  

The model is trained from scratch on 224×224 RGB images and includes standard data augmentation.

---

## Project Structure

```

.
├── main.py               # Trains the CNN and saves model_saved.h5
├── test_model.py         # Loads the saved model and tests it on a single image
└── v_data/
└── OCT2017/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── test/
├── CNV/
├── DME/
├── DRUSEN/
└── NORMAL/

````

Ensure the dataset is organized in this directory structure to work correctly with `ImageDataGenerator`.

---

## Requirements

Install the necessary dependencies:

```bash
pip install tensorflow keras numpy scipy
````

If using GPU acceleration, install the GPU-compatible version of TensorFlow.

---

## Model Overview

The model follows a simplified AlexNet architecture:

* Input shape: 224 × 224 × 3
* Five convolutional layers
* Max pooling after layers 1, 2, and 5
* Two fully connected layers (4096 units) with dropout
* Output layer: 4-class softmax

Training configuration:

* Optimizer: SGD
* Loss function: categorical crossentropy
* Metrics: accuracy

---

## Training

1. Place the dataset under `v_data/OCT2017/` following the structure shown earlier.

2. Adjust the training settings in `main.py` if required:

   ```python
   nb_train_samples = 5000
   nb_validation_samples = 90
   epochs = 22
   batch_size = 15
   ```

3. To start training:

   ```bash
   python main.py
   ```

After training completes, the model will be saved as:

```
model_saved.h5
```

---

## Testing and Inference

`test_model.py` provides a minimal example for testing a single image.

Example input used:

```python
image = load_img(
    'v_data/OCT2017/train/NORMAL/NORMAL-1384-1.jpeg',
    target_size=(224, 224)
)
```

Run the test script:

```bash
python test_model.py
```

The output includes:

* A printed model summary
* A prediction vector (softmax probabilities), for example:

  ```
  [0.01 0.05 0.10 0.84]
  ```

The order of classes is defined in `main.py`:

```
['CNV', 'DME', 'DRUSEN', 'NORMAL']
```

To test a different image, update the path in `test_model.py`.

---

## Notes

* Data augmentation used during training includes:

  * Rescaling
  * Shear
  * Zoom
  * Horizontal flipping
* `flow_from_directory` automatically infers labels from directory names.
* The project is designed to be easy to extend and modify.


