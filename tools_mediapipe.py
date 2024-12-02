import mediapipe as mp
import cv2
import numpy as np

def MediapipeDetection(image, model):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # image.flags.writable = False
  results = model.process(image)
  
  return results