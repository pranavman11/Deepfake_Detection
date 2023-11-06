# Deepfake Detection

Deepfakes are artificially manipulated media in which the face of a person in an existing image in the video is replaced with someone else’s face. They take advantage of powerful machine learning techniques to generate visual content with a high potential to deceive. 

We used the You-Look-Only-Once(YOLO) algorithm for object detection to extract faces (as objects) from the images. As we were training the model with YOLO, we had to continue with the whole implementation using the darknet framework.


The data on which we trained our model was collected from kaggle.com which had thousands of videos that were either fake or real (a Metadata JSON file was provided in order to initially distinguish fake and real videos). The video is required to be broken down into frames as to label each frame individually. A simple python script was used to break the video into frames which were stored in a folder. 

For labeling, Yolo label can be used which requires manually making a bounding box around the face which generates a .txt file of the same name as that of the image file which consists of a category of the frame(0 for fake, 1 for real), x-coordinate and y-coordinate of the center of the bounding box as well as height and width of the bounding box.
But this was a tedious task so we used dlib library (which is a library written in C++ and contains a lot of helpful machine learning algorithms) to detect the face(s) in the frames by running a frontal face detection function and get the category of the face from the metadata file and deriving the center coordinates and dimensions of the bounding box.

Training of the model was done on Google colab
The steps to train will be as follows:
- Connect Google Colab with Google Drive (where darknet folder is present)
- Check for CUDA compatibility.
- Load dataset and darknet folder to COLAB working space.
- Create a Symbolic link to drive, for storing weights.
- Compile darknet directory.
- Start the Training
- After this process, we get the output weights as .weight files

After the Google Colab has generated weights, we use those weights in a Python script which is given a video input and the weight file generated from the training process. Using these weights, we predict where there is a face in the current frame of not. If there is, we classify it as real or fake. The VideoWriter class helps to put all the frames into a single mp4 file with a  bounding box on each frame which has a face detected and the result whether it is fake or real.

Examples:

<img width="476" alt="Screenshot 2023-11-06 at 12 46 19 PM" src="https://github.com/pranavman11/Deepfake_Detection/assets/42564227/f4d1a71f-e053-47ac-92f5-f53ee6507889">
<img width="402" alt="Screenshot 2023-11-06 at 12 46 24 PM" src="https://github.com/pranavman11/Deepfake_Detection/assets/42564227/095a9bbb-190a-4fe3-a7ea-ccc93a50ece2">

The accuracy of the model was calculated using mean precision value metric and it came out to be ~70%
