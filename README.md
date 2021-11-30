# Synchrony Project

Repo for head nod detection for the Synchrony Project. 

## Obtaining OpenFace Data
The OpenFace test data can be found in the `data` folder and was generated using the `FeatureExtraction` executable from the Docker image found [here](https://hub.docker.com/r/algebr/openface/). More details can be found in the OpenFace GitHub [repository](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments).

## Head Nod Detection
There are two different head nod detection algorithms in this repo--the first uses optical flow and the second uses OpenFace facial landmakrs. 

### Optical Flow Using OpenCV
Optical flow often used in activity recognition and is used to estimate the object’s displacement vector caused by its motion or camera movements. Sparse optical flow, in which the motion vector for the specific set of objects/pixels is calculated, has been successfully applied to head pose estimation tasks. We use the Lucas-Kanade differential method of calculating the optical flow. Our implementation is coded using the OpenCV library and is able to take in a video or live webcam feed as an input. 

Our current approach is as follows: 
1. Each frame is pre-processed and converted to a black and white image. The algorithm then tries to detect a face within the video using a Haar Cascade Classifier. 
2. Once a face is found, we calculate the approximate center point of the face using the bounding rectangle produced by the Cascade Classifier. 
3. The optical flow of the center point is calculated. If the total displacement in the y-axis exceeds a defined threshold, then we classify this movement as a head nod. A threshold of 175 pixels was used to detect a head nod.

### OpenFace Facial Landmark Movements
The head nod detection primarily relies on peak detection using the 2D y-axis momements of the center of the nose (facial landmark #33 - for addtional details on OpenFace landmarks and outputs, see [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format)). A peak is estimated to be a single head nod. Head nod clusters can consist of multiple peaks/nods and are defined based on a threshold of how far distinct nods should be spaced out. A threshold of 2 seconds means that any peak that is 2 seconds from the last peak will be classified as a new, separate nod. 

After clusters of head nods are detected, the frequency of each nod is approximated using a fast Fourier transform. `headNod.py` returns the number of detected head nod clusters as well as the frequencies and duration of each nod cluster. 




 
