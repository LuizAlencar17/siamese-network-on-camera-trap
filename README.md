# Title

A Context-Aware Approach for Filtering Empty Images in Camera Trap Data Using Siamese Network

## What is it?

This repository is the implementation of the article "A Context-Aware Approach for Filtering Empty Images in Camera Trap Data Using Siamese Network". This article presents a method based on a Siamese Convolutional Neural Network (CNN) to filter empty images captured by camera traps. The proposed method takes into account information from the environment around the camera comparing the captured images with empty reference images obtained regularly from the same capture locations. Reference images are expected to highlight local vegetation features such as rocks, mountains and lakes. By calculating the similarity between the two images, the Siamese network determines whether or not the captured image contains an animal. We present a protocol for providing pairs of images to train the models, as well as the data augmentation techniques employed to improve the training procedure. Three different models of CNN are used as backbones for the Siamese network: MobileNetV2, ResNet50 and EfficientNetB0. In addition, experiments are conducted on three popular camera trap datasets: Snapshot Serengeti, Caltech, and WCS. The results demonstrate the effectiveness of the proposed method due to the capture location information considered, and its potential for fauna monitoring applications.

to run the pipeline for training or testing the models just use the following command:

```bash
python3 train.py --flagfile=configs/mobilenetv2_caltech_\(tag:no_serengeti_weights\)_256_siamese.config
```

or

```bash
python3 test.py --flagfile=configs/mobilenetv2_caltech_\(tag:no_serengeti_weights\)_256_siamese.config
```

note that there are settings to use with the serengeti database weights or without these weights. You can identify these settings through the filenames with "no_serengeti_weights" or "serengeti_weights".


if you want to run your own configuration, here are the descriptions of the parameters that you can customize:

```bash
--model_name="model name to be imported"
--dataset_name="dataset name to be used"
--tag="identified to use or not use weights from the serengeti dataset"
--model_type="model type, whether siamese or not"
--num_classes="number of classes to be used by models"
--image_size="image size"
--seed="seed to replicate results"
--num_epochs="number of epochs during model training"
--patience="number of how many times the model may not improve during training to make an early stop"
--batch_size="number of images per batch during training"
--train_filename="csv of images that will be used for training"
--val_filename="csv of images that will be used for validation"
--test_filename="csv of images that will be used for testing"
--images_path="directory of images that will be processed in training, testing and validation"
--images_path_ssd="directory of images that will be processed in training, testing and validation (if they are on an ssd)"
--checkpoint_path="location where templates will be saved"
--input_scale_mode="image preprocessing type"
```

## Datasets

In the directory ".whitelist/.scripts/create_whitelists.ipynb" has the script for creating pairs for models, including siamese and non-siamese models. Still in the directory ".whitelist/.scripts" there are scripts for you to copy images from hd to ssd, if you want.

In the file ".whitelist/.scripts/utils/map_description.json" there is a description of the parameters for you to use and create your own list of pairs of Siamese network, but if you want to use the same list used in our article you can download the .zip through the link DATASET_LINK.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)