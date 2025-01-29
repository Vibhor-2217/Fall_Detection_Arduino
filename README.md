# Fall_Detection_Arduino
This project uses YOLO pretrained model to train the model and then using opencv library it detects whether there is a fall or not. 

# Setting up on your device 
Step 1: Clone the repo 

Step 2: Change the path according to your device in dataset.yaml

Step 3: In the fall.py file, change the websocket host and port according to your device (you can also leave it the way it is)

Step 4: Open this link,(https://wokwi.com/projects/421389136161014785), change line 12 and 13. It will be the WiFi name and the password that you are connected to and most importantly keep the websocket server input the same as the one that you mentioned in your fall.py file. Changes to that might not make the program to run.  

1st run the fall.py file and then the  arduino compiled file. 

Note: The dataset can be obtained from Kaggle. The dataset which has been used here is Fall Detection Dataset. Link - https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset
