##
## SENSOR EFFECT AUGMENTATION PIPELINE 

This repository contains the tensorflow implementation of the Sensor Effect Augmentation Pipeline described in *Modeling Camera Effects to Improve Visual Learning from Synthetic Data* (https://arxiv.org/abs/1803.07721). This fork uses NumPy insted of obsolete TensorFlow.

### Setting up the Sensor Effect Pipeline
The pipeline of this repository was changed to use virtual env insted of docker, as it can simply run without gpu with proper libraries installed.
Simply create a new venv with ``` python -m venv sensorEnv ```, then install libraries listed in ``` requirements.txt ``` with ``` pip install -r requirements.txt ```.


### Running the Pipeline/Augmenting Images

To run the pipeline, use the command

```python main_aug.py ```

in the command line within the SensorEffectAugmentation folder.
You can customize the type of augmentations, the dataset to augmented, etc by modifying the ``` main_aug.sh```, which is described in more detail below.

The Sensor Effect Image Augmentation Pipeline is comprised of the following files:
  
	```
	usage: main.py [-h] [-n [N]] [-b [BATCH_SIZE]] [-c [CHANNELS]] [-i INPUT]
		       [-o [OUTPUT]] [--pattern [PATTERN]]
		       [--image_height [IMAGE_HEIGHT]] [--image_width [IMAGE_WIDTH]]
		       [--chromatic_aberration [CHROMATIC_ABERRATION]] [--blur [BLUR]]
		       [--exposure [EXPOSURE]] [--noise [NOISE]]
		       [--colour_shift [COLOUR_SHIFT]] [--save_params [SAVE_PARAMS]]

	Augment a dataset

	optional arguments:
	  -h, --help            show this help message and exit
	  -n [N]                sets the number of augmentations to perform on the
				dataset i.e., setting n to 2 means the dataset will be
				augmented twice
	  -b [BATCH_SIZE], --batch_size [BATCH_SIZE]
				size of batches; must be a multiple of n and >1
	  -c [CHANNELS], --channels [CHANNELS]
				dimension of image color channel (note that any
				channel >3 will be discarded
	  -i INPUT, --input INPUT
				path to the dataset to augment
	  -o [OUTPUT], --output [OUTPUT]
				path where the augmented dataset will be saved
	  --pattern [PATTERN]   glob pattern of filename of input images
	  --image_height [IMAGE_HEIGHT]
				size of the output images to produce (note that all
				images will be resized to the specified image_height x
				image_width)
	  --image_width [IMAGE_WIDTH]
				size of the output images to produce. If None, same
				value as output_height
	  --chromatic_aberration [CHROMATIC_ABERRATION]
				perform chromatic aberration augmentation
	  --blur [BLUR]         perform blur augmentation
	  --exposure [EXPOSURE]
				perform exposure augmentation
	  --noise [NOISE]       perform noise augmentation
	  --colour_shift [COLOUR_SHIFT]
				perform colour shift augmentation
	  --save_params [SAVE_PARAMS]
				save augmentation parameters for each image
	```
* ```main_aug.py```

	This is a master function that handles input flags and initializing the augmentation. 

* ```model_aug.py```

	This is a python module that defines the image augmentation class. It is called by ```main_aug.py```. To alter the sensor effect parameter ranges, you will need to change the values in the ```generate_augmentation``` function, lines 123-200 in this file.

* ```augmentationfunctions_tf.py```

	This is a python module that contains all of the sensor augmentation functions for blur, chromatic aberration, exposure changes, sensor noise, and color shifts.
	Please see our arxiv paper (https://arxiv.org/abs/1803.07721) for more information on these functions.

* ```geometric_transformation_module.py```

	This is a python module that implements affine warping; is used for augmenting images with chromatic aberration via warping the R and B color channels relative to the G channel.
	It is called by augmentationfunctions_tf.py

* ```pix2pix_lab2RGBconv.py```

	This is a python module that implements color conversion between RGB and LAB color spaces. It is called by augmentationfunctions_tf.py

Examples of GTA images augmented with the above sensor effects are located in ```Sensor_Augmentation_Pipeline/augFIGURES``` folder.




