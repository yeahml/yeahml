# Multi Segmentation

## The Data

The included dataset is very small (only 80 images total) and was downloaded for free from [here](http://www.mut1ny.com/face-headsegmentation-dataset). However, if someone, it would be possible to implement the included methodology for the entire dataset which is also available at the same website.

Note books for showing the steps necessary to 1) download the data 2) preprocess the data and 3) parse the data into tf records are shown in [./make_records](./make_records/) directory.

#### Sample Image

<p align="left">
<img src="./misc/image_and_mask_ex.png" alt="Example of a face image from the dataset and the corresponding output target" width="300">
</p>

Where the image on the left is of an example image and the image on the right is of the ground truth labeled image.

## Tensorboard Image Example

<p align="left">
<img src="./misc/tb_output_img.png" alt="Example of an image created in tensorboard of a sample face image, the segmentation prediction for the image throughout training, and the corresponding ground truth image" width="300">
</p>

Where the image on the left is of an example image and the image on the right is of the ground truth labeled image. The image second from the right is of the output during various stages of training (which can be viewed by adjusting the slide)