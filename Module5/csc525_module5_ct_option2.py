'''
MS - Artificial Intelligence and Machine Learning
Course: CSC525 - Principles of Machine Learning
Module 5: Critical Thinking Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
June 11, 2026

Problem statement: 
Option #2: Image Dataset Augmentation
State-of-the-art machine learning models use millions, sometimes billions, of parameters. 
When training our models, the number of distinct examples we need is proportional to the 
number of parameters our model has.

Invariance is a trait of a neural network model (for instance, an image classification model)
that can robustly classify objects even when the objects are placed in different orientations, 
have different sizes, and have illumination differences. Many of these potential differences 
in our dataset can be created artificially, instead of collecting more images. Methods could 
be to alter the brightness or contrast of the image, stretch or skew operations, or a variety
of translation methods.

For your assignment, submit a Python script that will take any image dataset and augment it 
in some way to expand the dataset. Submission must include a script that will augment 
any image files within its folder. 

Please include the un-augmented dataset with the augmented dataset and a short description 
of what was augmented.

https://datature.io/blog/image-augmentation-for-machine-learning-techniques-examples-code
'''

'''
needed installs:
(csc525) C:\Projs\Python\CSU\csc525>pip install Augmentor

'''


import os
import sys   #for command line arguments handling
import Augmentor


# Helper function.
# Clears the terminal
def clearScreen():
    if os.name == 'nt':  # For windows
        _ = os.system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = os.system('clear')
    return

# this fuinction creates the augumentation pipeline by calling the function: 
#       Augmentor.Pipeline(...) from the imported library Augmentor.
def create_pipeline(inputImagePath: str, outputImagePath: str, outputImageCount: int):
    # input validation
    if inputImagePath is None or len(inputImagePath.strip()) < 1:
        return
    if outputImagePath is None or len(outputImagePath.strip()) < 1:
        return
    inputImagePath = inputImagePath.strip()
    if len(os.listdir(inputImagePath)) < 1:
        return 
    outputImagePath = outputImagePath.strip()
    if outputImageCount < 1:
        return
    # input validation
    
    # 1. Create a pipeline pointing to the dataset folder    
    p = Augmentor.Pipeline(source_directory=inputImagePath, output_directory=outputImagePath, save_format="PNG")
   
    # 2. Add augmentation operations
    p.flip_left_right(probability=0.4)
    p.flip_top_bottom(probability=0.2)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5) # ??
    p.resize(0.3, 220, 250)
    p.skew(probability=0.5, magnitude=0.2)
    p.random_brightness(0.4, 0.5, 1.5)
    p.random_contrast(0.6, 0.5, 1.5)
        
    return p


# This is application execution main entry point, it calls above function to execute the needed job for this project.
# Then it calls the pipeline to create sample for defined number of output images like: p.sample(opImageCount).
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC525 - Principles of Machine Learning")
    print("Module 5: Critical Thinking Assignment")
    print("   Option #2: Image Dataset Augmentation\n")
    print(Augmentor.__version__) # display Augmentor version
    print("Valid format: (csc525) <your app path>>python csc525_module5_ct_option2.py <input images path> <output images path> <image count>")
    """
    orientations, sizes, stretch, skew, variety of translations, illumination, brightness, contrast.
    """
    # Input Arguments parsing
    # sys.argv[0] : python script file itself
    # sys.argv[1] : input images path
    # sys.argv[2] : output images path
    # sys.argv[3] : how many output images to create
    if sys.argv is None or len(sys.argv) != 4:
        print("Please run again with valid arguments")
        print("Valid format: (csc525) <your app path>>python csc525_module5_ct_option2.py <input images path> <output images path> <image count>")
    else:
        print("Arguments:", sys.argv) # displays the arguments
        opImageCount: int = 0
        try:
            opImageCount = int(sys.argv[3])  # how many output images to create
            if opImageCount > 0:
                p = create_pipeline(inputImagePath=sys.argv[1], outputImagePath=sys.argv[2], outputImageCount=opImageCount)
                p.sample(opImageCount)
        except ValueError:
            print("Invalid integer:", opImageCount)

# Test Executions:
# (csc525) C:\Projs\Python\csc525>python csc525_module5_ct_option2.py <input images path> <output images path, relative> <output images count>
# (csc525) C:\Projs\Python\csc525>python csc525_module5_ct_option2.py .\datafiles\module5 .\augment_output1 30
# (csc525) C:\Projs\Python\csc525>python csc525_module5_ct_option2.py .\datafiles\module5 .\augment_output2 20
