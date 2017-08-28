import numpy as np
from scipy import signal
from scipy import misc

def show_matrix(image, boundary):
    height = np.shape(image)[0]
    width = np.shape(image)[1]

    boundaries = np.copy(boundary)

    if np.max(boundaries[0]) == 0:
        boundaries[0] = boundaries[0]
    else:
        boundaries[0] /= np.max(boundaries[0])

    if np.max(boundaries[1]) == 0:
        boundaries[1] = boundaries[1]
    else:
        boundaries[1] /= np.max(boundaries[1])

    if np.max(boundaries[2]) == 0:
        boundaries[2] = boundaries[2]
    else:
        boundaries[2] /= np.max(boundaries[2])

    if np.max(boundaries[3]) == 0:
        boundaries[3] = boundaries[3]
    else:
        boundaries[3] /= np.max(boundaries[3])

    boundaries /= 2

    pixel_size = 5
    gap_size = 3
    padd = 4
    w = pixel_size+gap_size

    height_img = pixel_size*height + gap_size*(height-1) + padd
    width_img = pixel_size*width + gap_size*(width-1) + padd
    img = np.zeros((height_img,width_img,3))

    for i in range(height):
        for j in range(width):
            #scale and draw the images
            row = i*pixel_size + i*gap_size
            col = j*pixel_size +j*gap_size
            img[(row+2):(row+pixel_size+2),(col+2):(col+pixel_size+2),:] = image[i][j]

            #draw vertical top boundaries
            if i > 0:
                img[i*w, w*j+1:w*j+w, 1] += boundaries[0][i-1][j]
            if i < height-1:
                img[w*(i+1), w*j+1:w*j+w, 1] += boundaries[1][i+1][j]
            #draw horizontal boudaries
            if j > 0: #left
                img[w*i+1:w*i+w,w*j,2] += boundaries[2][i][j-1]
            if j < width-1: #right
                img[w*i+1:w*i+w, w*(j+1), 2] += boundaries[3][i][j+1]

    # plt.imshow(img)
    # plt.show()
    return img