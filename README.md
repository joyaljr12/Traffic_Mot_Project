# Traffic_Mot
Traffic flow analysis from traffic junction camera

We are taking in feed from a youtube livecamera video and passing it to the detection module. The module uses a pretrained yolov4 object detection model to detect the position and class of the vehicles . This information from multiple frame is taken in by the Deepsort algorithm to create a track and unique ID for each vehicle in the video.
The direction detection algorithm also extract the information of the direction travelled by the vehicle along with the time of detection ,store it in form of an excel sheet.

The software is optimised to run on Jetson devices.(TensorFlow , TensorFlow Lite and TensorRT)

# Software Flow Chart

<img width="398" alt="image" src="https://user-images.githubusercontent.com/102171203/159943538-9750baaf-4daf-4c1f-8355-7476533fc2fd.png">



# Detection Sample

<img width="210" alt="image" src="https://user-images.githubusercontent.com/102171203/159945316-b2a3b3b8-8178-40a5-bfa0-a73221126e18.png">       <img width="274" alt="image" src="https://user-images.githubusercontent.com/102171203/159945344-f9754bb3-e10c-4543-a9f0-e973fe7d5375.png">

Tracking Module Output with same ID assigned to vehicles across junction
