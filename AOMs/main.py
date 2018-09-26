# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:40:51 2018

@author: hungl
"""


import menpo.io as mio
from menpo.visualize import print_progress
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68_trimesh

path_to_images = '/path/to/lfpw/trainset/'
training_images = []
for img in print_progress(mio.import_images(path_to_images, verbose=True)):
    # convert to greyscale
    if img.n_channels == 3:
        img = img.as_greyscale()
    # crop to landmarks bounding box with an extra 20% padding
    img = img.crop_to_landmarks_proportion(0.2)
    # rescale image if its diagonal is bigger than 400 pixels
    d = img.diagonal()
    if d > 400:
        img = img.rescale(400.0 / d)
    # define a TriMesh which will be useful for Piecewise Affine Warp of HolisticAAM
    labeller(img, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)
    # append to list
    training_images.append(img)
    
#    
#import matplotlib.pyplot as plt
#
## Load and convert to grayscale
#image = mio.import_image(path_to_lfpw / 'image_0152.png')
#image = image.as_greyscale()
#
## Detect face
#bboxes = detect(image)
#
## Crop the image for better visualization of the result
#image = image.crop_to_landmarks_proportion(0.3, group='dlib_0')
#bboxes[0] = image.landmarks['dlib_0'].lms
#
#if len(bboxes) > 0:
#    # Fit AAM
#    result = fitter.fit_from_bb(image, bboxes[0], max_iters=[15, 5],
#                                gt_shape=image.landmarks['PTS'].lms)
#    print(result)
#
#    # Visualize
#    plt.subplot(131);
#    image.view()
#    bboxes[0].view(line_width=3, render_markers=False)
#    plt.gca().set_title('Bounding box')
#
#    plt.subplot(132)
#    image.view()
#    result.initial_shape.view(marker_size=4)
#    plt.gca().set_title('Initial shape')
#
#    plt.subplot(133)
#    image.view()
#    result.final_shape.view(marker_size=4, figure_size=(15, 13))
#    plt.gca().set_title('Final shape')