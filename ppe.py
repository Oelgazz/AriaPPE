# PPE.py
# Author: Omar Elgazzar
# Aria Technologies

# Simple program that uses a YOLO model to identify
# violations in PPE. If a violation occurs, an email alert
# containing the type of violation (ex: not wearing a helmet)
# and an image hilighting that violation is sent to the configured
# email.

# Command line arguments:   'model file' 'threshold' 'server email' 'server password' 'receiver email'

#       Example: "python ppe.py best.pt 10 server@email.com password alert@email.com"


from ultralytics import YOLO
import cv2
import os
import smtplib
from email.message import EmailMessage
import sys

# Function that sends an email noting the type of violation and an attachment of the image
def sendMessage(violationType, model, image, server, sender, receiver):

    # NOTE: "show=True" is used for debugging. To be changed to "False" when done 
    model.track(source=image, show=False, conf=0.3, save=True, show_conf=False) # Saves image to ./runs/detect/track/image0.jpg

    # Set up email
    message = EmailMessage()
    message['Subject'] = "A Violation has Occured on Your Site"
    message['From'] = sender
    message['To'] = receiver
    message.set_content("A worker on your site was found violating PPE. This worker was not wearing "+ violationType
                       +". Please take necessary action.\n\nFor confirmation, see the image attached.")
    
    # Read saved image and attach it to email
    with open("runs/detect/track/image0.jpg", 'rb') as fp:
        data = fp.read()
    message.add_attachment(data, maintype='image', subtype='jpeg')

    # Send the email
    server.send_message(message)

###         MAIN         ###

if len(sys.argv) != 6:
    print("Error: Invalid number of arguments")
    exit()

# Get model file
modelFile = sys.argv[1]

# Get threshold for number of violations
threshold = int(sys.argv[2])

# Get server's email
sender = sys.argv[3]

# Get server's password
password = sys.argv[4]

# Get reciever's email
reciever = sys.argv[5]

# Set up server
s = smtplib.SMTP('smtp-mail.outlook.com', 587)
s.starttls()
s.login(sender, password)

try:
    # Setup YOLO model
    model = YOLO(modelFile)

    # Path to video file 
    vidObj = cv2.VideoCapture(0)

    # Counters of number of frames with violations
    maskViolation = 0
    helmetViolation = 0
    goggleViolation = 0
    gloveViolation = 0
  
    # checks whether frames were extracted 
    success = 1

    # Loop through each image
    while success:

        # Get an image
        success, image = vidObj.read()

        # Run the image through the model
        results = model.track(source=image, conf=0.3)

        # If there is a object defined as a violation, increase that violation's count
        if results[0].boxes.cls.tolist().count(7) > 0:
            maskViolation += 1

        if results[0].boxes.cls.tolist().count(6) > 0:
            helmetViolation += 1

        if results[0].boxes.cls.tolist().count(5) > 0:
            goggleViolation += 1

        if results[0].boxes.cls.tolist().count(4) > 0:
            gloveViolation += 1
    

        # For each violation type, check if the count surpassed the threshold
        #   If so, send an alert

        if maskViolation > threshold:
            sendMessage("MASK", model, image, s, sender, reciever)
            maskViolation = 0
        
            # NOTE: Debugging Code for each elif statement (To be removed)
            #ex = input("Do you wish to stop the program? (y/n): ")
            #if (ex == "Y" or ex == "y"):
            #    break

        elif helmetViolation > threshold:
            sendMessage("HELMET", model, image, s, sender, reciever)
            helmetViolation = 0

        elif goggleViolation > threshold:
            sendMessage("GOGGLES", model, image, s, sender, reciever)
            goggleViolation = 0
        
        elif gloveViolation > threshold:
            sendMessage("GLOVES", model, image, s, sender, reciever)
            gloveViolation = 0

    # If the program were to exit, reset.
    os.remove("runs/detect/track/image0.jpg")
    os.removedirs("runs/detect/track")
    s.quit()


# If an error occurs, print the error and reset
except Exception as e:
    os.remove("runs/detect/track/image0.jpg")
    os.removedirs("runs/detect/track")
    s.quit()
    print(e)