# Commercial-Aircraft-Information-Extractor
This is my final project I worked on for EECS 4422-Computer Vision during Fall of 2019. <br>
Please check project_report.pdf for detailed information.

Requirements:
  NVIDIA GPU <br>
  NVIDIA Drivers <br>
  CUDA <br>
  cuDNN <br>
  Python 3+
  Everything in requirements .txt <br>
  
  You can run <pip3/pip> install -r requirements.txt to install all the required packages.
  
  In addition, download the following link and unzip its contents to the same directory as the contents of this repo.<br>
  https://drive.google.com/file/d/1OTVe0L6s5-TLLrBFpv7C6SldD77nLQ_w/view?usp=sharing <br>
  
  The coco model for Mask R-CNN will be downloaded automatically on the first run. <br>
  <b>THE DOWNLOAD IS ROUGHLY ~250 MB YOU HAVE BEEN WARNED.</b><br>
 
 How to Run: <br>
  Simply run extract_information.py choosing the correct arguments. <br>
 
 Arguments: <br>
    --single_image : Choose this option with --image_path to work on a single image file. <br>
    --multiple_images : Choose this option with --images_path to work on multiple images. <br>
    --image_path : Path for the image file. Only specify if you chose --single_image. <br>
    --images_path : Path for the folder containing image folders. Only specify if you chose --multiple_images. <br>
    --verbose : Choose this option to have intermediary images saved. <br>
 
 Example Running Commands: <br>
  <python/python3> extract_information.py --single_image --image_path images/tc-jjz.jpg <br>
  <python/python3> extract_information.py --single_image --image_path images/tc-jjz.jpg --verbose <br>
  <python/python3> extract_information.py --multiple_images --images_path images <br>
  <python/python3> extract_information.py --multiple_images --images_path images --verbose <br>
  
  For convenience, if you create a directory with the name "images" and put your input images in it then you can omit the --images_path argument.
