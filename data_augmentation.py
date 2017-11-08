import cv2
import random

# IMAGE_SIZE = 200
# images = []
# img = cv2.imread('sintel1.png')
# img2 = cv2.imread('test.jpg')
#
# img = cv2.resize(img, (200,200))
# img2 = cv2.resize(img2, (200,200))
#
# images.append(img)
# images.append(img2)


def flip_operation(images):
    ''' Takes a list of frames and flips all the frames either about the
    horizontal axis or the vertical axis and returns the list of
    augmented_images'''
    augmented_images = []
    randint = random.randint(0,1)
    print randint
    for image in images:
        augmented_images.append(cv2.flip(image,randint))

    return augmented_images

def crop_images(images):
    ''' Takes a list of frames and generates a randint(0,5) based on whose value
    it will translate the image 20 percent in one of the 4 direactions and
    returns the list of augmented_images'''

    randint = random.randint(1,5)
    # radinint = 5 , do nothing , randint = 1 , remove bottom 20 percent
    # randint = 4, remove top 20 percent, randint = 2 , remove right 20 percent
    # randint = 3. remove left 20 percent
    print randint
    augmented_images = []
    #Saving the size of the image
    x1=1        #Doubt 1 or 0
    y1=1        #Doubt 1 or 0
    x2=images[0].shape[1]
    y2=images[0].shape[0]
    if randint == 5:
        return images
    else:
        if randint == 1 :
            y2 = int(0.8*y2)
        elif randint == 2 :
            x2 = int(0.8*x2)
        elif randint==3:
            x1 = int(0.2*x2)
        elif randint==4:
            y1 = int(0.2*y2)
        for image in images:
            temp_img = image[y1:y2, x1:x2]
            augmented_images.append(cv2.resize(temp_img, (200,200)))
        return augmented_images;

def augment_data(frames):
    '''wrapper function which takes all the frames and randomly selects a
    certain percentage of frames and either flips or crops the selected frame
    and the next five frames'''
    flip_percentage = 0.4
    crop_percentage = 0.4

    randint = random.randint(0,1)
    if randint == 0:
        return flip_operation(frames)
    else
        return crop_images(frames)


# end_images = crop_images(images)
# cv2.imshow("a1",images[0])
# cv2.imshow("b1",images[1])
# cv2.imshow("1",end_images[0])
# cv2.imshow("2",end_images[1])
# cv2.waitKey(0)
