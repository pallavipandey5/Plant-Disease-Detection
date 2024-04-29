# Plant Disease Detection with MaskRCNN

This repository contains code for training and evaluating a plant disease detection model using the MaskRCNN architecture with the Detectron2 library. The model is trained on a dataset consisting of 1345 images, with 1000 images used for training and 345 images for testing. Additionally, augmented images have been included in the training process.

## Files

- **train.ipynb**: Jupyter notebook for training the plant disease detection model using the provided dataset. It utilizes 1000 images for training, with augmented images included, and evaluates the model's performance on the remaining 345 test images. The dataset is stored in the `all_train` and `all_test` directories along with their respective JSON files.
  
- **test.ipynb**: Jupyter notebook for training the model using the MaskRCNN architecture with the Detectron2 library.

- **test_instance.py**: Python script for testing the trained model on an instance.

- **visualize_training.py**: Python script for visualizing the masking on training images based on ground truth annotation.

- **IoU.py**: Python script for calculating Intersection over Union (IoU) metrics.

- **binary_classification.py**: Python script for performing binary classification before disease detection.

- **SOTA_evaluation.py**: Python script for evaluating the model's performance using state-of-the-art metrics including F1 score, precision, and recall.

- **custom_evaluation.py**: Python script for custom evaluation method. Images receive a score of 1 if they have at least 20% correct predictions with a threshold of 0.4. The total score is calculated as the number of images with a score of 1 divided by the total number of images in the test set.

- **requirements.txt**: Text file specifying the environment requirements for running the code.

## Instructions

1. Clone the repository to your local machine:

    ```bash
    git clone <repository_url>
    ```

2. Set up the environment using the provided `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

3. Follow the instructions in the `train.ipynb` and `test.ipynb` notebooks to train and evaluate the plant disease detection model.

4. Utilize the provided scripts (`test_instance.py`, `visualize_training.py`, `IoU.py`, `binary_classification.py`, `SOTA_evaluation.py`, `custom_evaluation.py`) for additional testing, evaluation, and analysis.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The dataset used in this project is from [insert dataset source].
- The MaskRCNN architecture is implemented using the [Detectron2 library](https://github.com/facebookresearch/detectron2).
- Special thanks to [insert names] for their contributions to this project.
