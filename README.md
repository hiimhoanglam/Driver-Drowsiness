# Project
Driver Drowsiness Detection System - Final Project for Introduction to Artificial Intelligence course

# Model Pipeline
Construct dataset from online sources
Face detection using Haar Cascade <br/>
Landmark detection using Dlib <br/>
From landmark => Extract keypoints => Eyes region <br/>
ViT model fine-tuned with our dataset <br/>
Eyes => ViT => Classification(Drowsy/Non-drowsy) <br/>
If 5 consecutive frames classified as Drowsy => Warning display + Play sound alarm
![image](https://github.com/user-attachments/assets/ed87dcd0-24d6-471f-8dc8-6ef5fc64245f)

# Model training results
![image](https://github.com/user-attachments/assets/4173232d-9586-4797-9006-28af26d64de7)

# Qualitative Results
![image](https://github.com/user-attachments/assets/042d1b81-ee5a-4057-a119-95d512fc3be4)

# Quantitive Results
![image](https://github.com/user-attachments/assets/2d78641b-1eeb-4643-96f4-8e4def15ccfa)
![image](https://github.com/user-attachments/assets/e5e40701-6a08-44ac-9dc8-3c01407ffc39)


