# A Context-Aware Approach for Filtering Empty Images in Camera Trap Data Using Siamese Network

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

In the file ".whitelist/.scripts/utils/map_description.json" there is a description of the parameters for you to use and create your own list of pairs of the Siamese network, but if you want to use the same list used in our article you can download the . zip through the link [Datasets](https://drive.google.com/drive/folders/1vIp3HSgzngNDg1_GtSgRjFgTwb9Z8k5Z?usp=sharing). In case you want to use exactly the same models that we generated with exactly the same weights and parameters, you can download them through this link [Models](https://drive.google.com/drive/folders/1vIp3HSgzngNDg1_GtSgRjFgTwb9Z8k5Z?usp=sharing)

For you to acquire the images of the datasets used, we suggest you download them from [Lila Science](https://lila.science/datasets) because there you will find everything you need which are the images and metadata of each dataset

If you wanted to run the script for creating image pairs for the Siamese network, just configure the following parameters in the ".whitelist/.scripts/utils/map.json" file:

```bash
datasets: ["wcs", "wellington", "serengeti", "caltech"]

{
    "datasets": {
            "metadata": "location of the json file where the dataset metadata is",
            "images_path": "location of the directory where the dataset images are located",
            "images_path_ssd": "location of the directory where the dataset images are located, only if you have ssd and want to transfer the images from hd to ssd",
            "images_filenames_glob_*": "the subfolder structure of the directory where the images are located",
            "path_target_time_siamese": "location where the .csv with the siamese image pairs should be created",
            "path_target_time": "location where the .csv with the non-siamese image pairs should be created",
            "img_format": "image format"
    },
    "models": {
        "zilong": {
                "path_output_animal": "location where the list of images with animals will be saved",
                "path_output_empty": "location where the list of images without animals will be saved",
                "path_csv": "location where the .csv with the classification results will be saved",
                "path_tmp": "location of temporary directory for running zilong (in this directory little control information is stored)",
                "path_test": "location from where the list of images will be loaded to be sorted",
                "path_results": "location where standard zilong outputs will be saved",
                "command": "shell command to run zilong"
        }
    }
}
```

## Models

The figure below shows the difference structurally of the Siamese and non-Siamese models

![alt text](https://github.com/LuizAlencar17/siamese-network-on-camera-trap/blob/main/.readme_files/our_method_en.jpg?raw=true)

## Results

The figures below show the performance of the Siamese and non-Siamese models in the three databases used.

![alt text](https://raw.githubusercontent.com/LuizAlencar17/siamese-network-on-camera-trap/69b73a8fa63b4835ef36d657c0005573730e4ac2/.readme_files/accuracy_of_models_in_the_caltech_database_en.svg)


![alt text](https://raw.githubusercontent.com/LuizAlencar17/siamese-network-on-camera-trap/69b73a8fa63b4835ef36d657c0005573730e4ac2/.readme_files/accuracy_of_models_in_the_serengeti_database_en.svg)


![alt text](https://raw.githubusercontent.com/LuizAlencar17/siamese-network-on-camera-trap/69b73a8fa63b4835ef36d657c0005573730e4ac2/.readme_files/accuracy_of_models_in_the_wcs_database_en.svg)


## Camera Trap

Camera traps are heat- or motion-activated cameras placed in natural environments to monitor and investigate animal populations and behavior. They are used to locate endangered species, identify important habitats, monitor places of interest and analyze patterns of wildlife activity. These devices are capable of capturing tens of thousands of images. However, the extraction of information from these images is traditionally performed manually. The number of people available to extract this information is extremely limited compared to the amount of images generated. For this reason, much of the valuable knowledge contained in the data repositories of these images remains untapped.

![alt text](https://raw.githubusercontent.com/LuizAlencar17/siamese-network-on-camera-trap/69b73a8fa63b4835ef36d657c0005573730e4ac2/.readme_files/serengeti_day_stage_en.jpg)

![alt text](https://raw.githubusercontent.com/LuizAlencar17/siamese-network-on-camera-trap/69b73a8fa63b4835ef36d657c0005573730e4ac2/.readme_files/serengeti_year_stage_en.jpg)

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Citation

If you find this code useful in your research, please consider citing:

```bash
@InProceedings{luiz_2023_sibgrapi,
    author    = {Alencar, Luiz, and Cunha, Fagner and dos Santos, Eulanda M. },
    title     = {A Context-Aware Approach for Filtering Empty Images in Camera Trap Data Using Siamese Network},
    booktitle = {Proceedings of the IEEE/SIBGRAPI - Conference on Graphics, Patterns and Images},
    month     = {August},
    year      = {2023}
}
```

## Contact

If you have any questions, feel free to contact Luiz Alencar (e-mail: fabio.alencar644@gmail.com) or Github issues.

## License

[MIT](https://choosealicense.com/licenses/mit/)