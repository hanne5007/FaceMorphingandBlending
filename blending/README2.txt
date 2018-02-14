#README.txt
#Author: Hantian Liu
#Project: Blending

1. run_blend.py:
Run run_blend.py would show the source image first, which needs to click out a mask, using maskImage.py. And then it calls the seamlessCloningPoisson.py. With the result, it would generate my own blended image, shown in Blending_output_CatBuilding.py.
Commented part would generate the Blending_output_sample.png, which is the blended sample image given on Piaza.
To generate new blending images, the filename for source (simg) and target images (timg) needs to be changed. Also, adjust the resizing and offset parameters if necessary. 

3. maskImage.py:
It calls drawmask.py inside. 

4. reconstructImg.py:
It calls getCoefficientMatrix.py inside. 