# Imports
import numpy

# Function 1: Take filter parameters and build 2 oriented filters with different polarities for connection pattern from the LGN to V1
# Usage : filters1, filters2 = createFilters(numOrientations=8, size=4, sigma2=0.75, Olambda=4)
def createFilters(numOrientations, size, sigma2, Olambda, phi):

    # Initialize the filters
    filters1 = numpy.zeros((numOrientations, size, size))
    filters2 = numpy.zeros((numOrientations, size, size))

    # Fill them with gabors
    midSize = (size-1.)/2.
    maxValue = -1
    for k in range(0, numOrientations):
        theta = numpy.pi * (k + 1) / numOrientations + phi
        for i in range(0, size):
            for j in range(0, size):
                x = (i - midSize) * numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
                y = -(i - midSize) * numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
                filters1[k][i][j] = numpy.exp(-(x*x + y*y)/(2*sigma2)) * numpy.sin(2*numpy.pi*x/Olambda)
                filters2[k][i][j] = -filters1[k][i][j]
                

    # Rescale filters so max value is 1.0
  #  for k in range(0, numOrientations):
  #      maxValue = numpy.amax(numpy.abs(filters1[k]))
  #      filters1[k] /= maxValue
  #      filters2[k] /= maxValue
  #      filters1[k][numpy.abs(filters1[k]) < 0.3] = 0.0
  #      filters2[k][numpy.abs(filters2[k]) < 0.3] = 0.0
        
    # Normalize filters
    sumxx = numpy.zeros(numOrientations)
    for k in range(0, numOrientations):
        for i in range(0, size):
            for j in range(0, size):
                sumxx[k] += filters1[k][i][j] * filters1[k][i][j]
        filters1[k] /= sumxx[k]    	
        filters2[k] /= sumxx[k]    	

 #   for k in range(0, numOrientations):
 #       print(k)
 #       print(filters1[k])
 #       print(filters2[k])
            
    return filters1, filters2


# Function 2: Take filter parameters and build connection pooling and connection filters arrays
# Usage (for V1 e.g.) : V1poolingfilters, V1poolingconnections1, V1poolingconnections2 = createPoolConnAndFilters(numOrientations=8, VPoolSize=3, sigma2=4.0, Olambda=5)
# Usage (for V2 e.g.) : V2poolingfilters, V2poolingconnections1, V2poolingconnections2 = createPoolConnAndFilters(numOrientations=8, VPoolSize=7, sigma2=26.0, Olambda=9)
def createPoolingConnectionsAndFilters(numOrientations, VPoolSize, sigma2, Olambda, phi):

    # Build the angles in radians, based on the taxi-cab distance
    angles = []
    for k in range(numOrientations/2):
        taxiCabDistance = 4.0*(k+1)/numOrientations
        try:
            alpha = numpy.arctan(taxiCabDistance/(2 - taxiCabDistance))
        except ZeroDivisionError:
            alpha = numpy.pi/2 # because tan(pi/2) = inf
        angles.append(alpha + numpy.pi/4)
        angles.append(alpha - numpy.pi/4)

    # This is kind of a mess, but we could code it better
    for k in range(len(angles)):
        if angles[k] <= 0.0:
            angles[k] += numpy.pi
        if numOrientations == 2: # special case ... but I could not do it otherwise
            angles[k] += numpy.pi/4

    # Sort the angles, because they are generated in a twisted way (hard to explain, but we can see that on skype)
    angles = numpy.sort(angles)

    # Set up orientation kernels for each filter
    midSize = (VPoolSize-1.0)/2.0
    VPoolingFilters = numpy.zeros((numOrientations, VPoolSize, VPoolSize))
    for k in range(0, numOrientations):
        theta = angles[k] + phi
        for i in range(0, VPoolSize):
            for j in range(0, VPoolSize):

                x =  (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
                y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
                if numpy.abs(x) < 0.1:
                    VPoolingFilters[k][i][j] = 1.0

    # Set layer23 pooling connections (connect to points at either extreme of pooling line ; 1 = to the right ; 2 = to the left)
    VPoolingConnections1 = VPoolingFilters.copy()
    VPoolingConnections2 = VPoolingFilters.copy()

    # Do the pooling connections
    for k in range(0, numOrientations):

        # want only the end points of each filter line (remove all interior points)
        for i in range(1, VPoolSize - 1):
            for j in range(1, VPoolSize - 1):
                VPoolingConnections1[k][i][j] = 0.0
                VPoolingConnections2[k][i][j] = 0.0

        # segregates between right and left directions
        for i in range(0, VPoolSize):
            for j in range(0, VPoolSize):
                if j == (VPoolSize-1)/2:
                    VPoolingConnections1[k][0][j] = 0.0
                    VPoolingConnections2[k][VPoolSize-1][j] = 0.0
                elif j < (VPoolSize-1)/2:
                    VPoolingConnections1[k][i][j] = 0.0
                else:
                    VPoolingConnections2[k][i][j] = 0.0


 #   for k in range(0, numOrientations):
#		print k
#		print VPoolingFilters[k]
#		print VPoolingConnections1[k]

    return VPoolingFilters, VPoolingConnections1, VPoolingConnections2