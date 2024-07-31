# PPE Violation Detection

PPE.py is a program that uses YOLO to detect whether workers are violating PPE principles. It uses real-time video and counts the number of frames that has a violation.
If that frame count exceeds a defined threshold, the program sends an alert via email.

In this document we will go over the main flow of the program, how to run the program, and the steps taken to set up YOLO and the data set.

## Program Flow

The script starts off by initializing the YOLO model using the passed in file, `modelFile`:

```python
model = YOLO(modelFile)
```

After that the program initializes the camera input using cv2's `VideoCapture()` function and initializes the violation counters.

Going into the main loop, an image is extracted from the camera and fed into the model:

```python
results = model.track(source=image, conf=0.3)
```

From the results we check if there are any violations. This is done by checking if the model classified an object as a violation (no_mask, no_helmet, etc.). If there is a violation, the corresponding counter is incremented.

When a counter exceeds the threshold, an email alert is sent and the counter is reset. To send the email, we start with saving the image showing the violation to "/runs/detect/track". We then specify the message's subject, sender, receiver, and content. Finally, we attach the image and send the message.

This program is designed to loop constantly. If an error were to occur or the loop were to exit, the program resets by closing the email server and the images are deleted.

## Running the Program

This script is a command line script, meaning that in order to run the program you need to enter a command into the terminal. First, ensure that the ".pt" model file and the program are in the same folder. After that open the command prompt/terminal and navigate to that folder.

To run the program, enter the following command into the terminal:

```python ppe.py 'model' 'threshold' 'server email' 'server password' 'receiver email'```

Replacing 'model' with the name of the ".pt" model file, 'threshold' with the number of frames with violations needed to send an alert, 'server email' and 'server password' with the email and password to send the alert from, and 'receiver email' with the email to send the alert to.

WARNING: If you were to end the program using the "Keyboard Interrupt", you will need to delete all folders under "/runs/detect" manually to ensure that the images sent in the alert updates.

## Setting up the Dataset

The dataset we used for this application was a data set on roboflow. It can be found here: https://universe.roboflow.com/mohamed-traore-2ekkp/ppe-detection-l80fg

You can use other datasets and models with this script, but ensure that the model passed in is of ".pt" format.
