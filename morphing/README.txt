#README.txt
#Author: Hantian Liu
#Project: Morphing

1. run.py:
Run run.py would generate the Morphing_output_backto2016.gif, where it morphs my face in 2017 (image1.jpg) to my face in 2016 (image2.jpg), using the saved mat of correspondences. 
Another result Morphing_output_Me to Amber Heard.gif is generated from morphing the photo of me (image1.jpg) to Amber Heard (image3.jpg).  
To generate new morphing gifs using run.py, it needs to change the filename for starting (img1) and ending images (img2). And correspondences could be picked using click_correspondences (commented). 

3. click_correspondences.py: 
Input is needed for the number of correspondence points in both images when calling click_correspondences. Note that boundary points are ignored when choosing correspondences, since I include all necessary boundary points later in morph_tri.py. 
