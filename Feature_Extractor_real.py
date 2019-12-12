# Import packages
import numpy as np
import pandas as pd
import cv2
import math

class Feature_Extractor_Real:
    # class defines feature extraction methods for extracting features out of an image
    def __init__(self, bilateralFilter_diameter = 100, bilateralFilter_sigmaColor = 500, bilateralFilter_sigmaSpace = 10, 
                 adaptiveThreshold_blockSize = 801, adaptiveThreshold_C = 10, borderMargin = 50):
        self.bilateralFilter_diameter = bilateralFilter_diameter
        self.bilateralFilter_sigmaColor = bilateralFilter_sigmaColor
        self.bilateralFilter_sigmaSpace = bilateralFilter_sigmaSpace
        self.adaptiveThreshold_blockSize = adaptiveThreshold_blockSize
        self.adaptiveThreshold_C = adaptiveThreshold_C
        self.borderMargin = borderMargin

    def img_width(self, img):
        return img.shape[1]
    
    def img_height(self, img):
        return img.shape[0]
    
    def count_level(self, img):
        # Function that counts the number of floors
        
        # convert to grayscale
        gray_real = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray_real.shape
        # Get width of the image
        width = shape[1]
        height = shape[0]
        size = max(shape[0],shape[1])

        # bilateral filtering
        img_filtered = cv2.bilateralFilter(gray_real, self.bilateralFilter_diameter, self.bilateralFilter_sigmaColor, self.bilateralFilter_sigmaSpace)
        # ret, thresh_real = cv2.threshold(img_filtered, 75, 255, 0)
        # adaptive thresholding
        thresh_real = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptiveThreshold_blockSize, self.adaptiveThreshold_C)
        # contour detection
        contours, h = cv2.findContours(thresh_real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate polygons from contours
        approx_polygons = [cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) for contour in contours]
        # area of the image
        area = thresh_real.shape[0]*thresh_real.shape[1]
        # filter contours which have area>0.1% of image area
        polygons = [polygon for polygon in approx_polygons if cv2.contourArea(polygon) > 0.001*area]
        # get convex hulls of selected polygons 
        convex_polygons = [cv2.convexHull(polygon) for polygon in polygons if len(polygon)]
        # filter by number of sides (between 3 and 6) and area of polygons (between 0.1% and 15% of the image area)
        final_polygons = [cv2.convexHull(polygon) for polygon in convex_polygons if len(polygon) >= 3 and len(polygon) <= 6 
                          and cv2.contourArea(polygon) > 0.001*area and cv2.contourArea(polygon) < 0.15*area]    

        ## remove polygons which are too close to the borders
        redflag = [0]*len(final_polygons)
        for i in range(len(final_polygons)):
            q = final_polygons[i]
            for point in q:
                if abs(point[0][0] - gray_real.shape[1]) < 50 or abs(point[0][0]) < self.borderMargin:
                    redflag[i] = 1
                if abs(point[0][1] - gray_real.shape[0]) < 50 or abs(point[0][1]) < self.borderMargin:
                    redflag[i] = 1
        ## final openings
        openings = [final_polygons[i] for i in range(len(final_polygons)) if redflag[i] != 1]

        ## get lines from openings
        lines = []
        flags = []
        if len(openings)!= 0:        
            for i in range(len(openings)):
                polygon = openings[i]
                for point_id in range(len(polygon)):
                    x1 = polygon[point_id-1][0][0]
                    y1 = polygon[point_id-1][0][1]
                    x2 = polygon[point_id][0][0]
                    y2 = polygon[point_id][0][1]
                    lines.append(np.array([[x1,y1,x2,y2]], dtype='int32'))  

            # Delete lines which are very vertical
            flags = [0]*len(lines)  # flags will mark the redundant lines as 1, lines we need as 0
            for i in range(len(lines)):
                if (abs(lines[i][0][0]-lines[i][0][2])) == 0: # vertical lines
                    flags[i] = 1 
                elif (abs(lines[i][0][1]-lines[i][0][3])/abs(lines[i][0][0]-lines[i][0][2]) > 1):  # lines with absolute slope>1
                    flags[i] = 1

        ## filtered lines
        lines_filtered = []     
        if len(lines)!=0:
            for i in range(len(lines)):
                if (flags[i] == 0):
                    lines_filtered.append(lines[i])

        # draw the obtained lines on a blank canvas in order to use sophisticated opencv HoughLinesP function    
        edges = cv2.cvtColor(np.zeros(shape, dtype='uint8'), cv2.COLOR_GRAY2BGR)
        if len(lines_filtered)!=0:
            for i in range(len(lines_filtered)):
                line = lines_filtered[i]
                x1,y1,x2,y2 = line[0]
                cv2.line(edges, (x1,y1), (x2,y2), (255,255,255), int(0.01*size))
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

        # Detect lines representing the floors which are longer than 1% of the width of the building
        lines_processed = cv2.HoughLinesP(edges, rho = 1, theta = math.pi/180, threshold = 10, minLineLength = 0.01*width, maxLineGap = width)

        lines_final = [] 
        flags = []
        if (type(lines_processed) != type(None)):
            flags = [0]*len(lines_processed)  # flags will mark the redundant lines as 1, lines we need as 0
            for i in range(len(lines_processed)):
                # Delete lines which are very small or are very vertical or are very close to each other
                line = lines_processed[i]
                if (abs(line[0][0]-line[0][2])) == 0: # vertical lines
                    flags[i] = 1 
                elif (abs(line[0][1]-line[0][3])/abs(line[0][0]-line[0][2]) > 0.25):  # lines with absolute slope>1
                    flags[i] = 1
                if (abs(line[0][0]-line[0][2])**2 + abs(line[0][1]-line[0][3])**2)**0.5 < 0.1*size: # lines with small length
                    flags[i] = 1
                for j in range(len(lines_processed)):
                    if j < i and (abs(line[0][1]-lines_processed[j][0][1]) + abs(line[0][3]-lines_processed[j][0][3]) <  0.15*height):  # detect lines very close to each other (vertically) 
                        flags[j] = 1
        
        # counter keeps track of number of levels
        counter = 1
        lines_final = []     
        if (type(lines_processed) != type(None)):
            for i in range(len(lines_processed)):
                if (flags[i] == 0):
                    counter = counter+1
                    lines_final.append(lines_processed[i])
        # adjustment for overestimation errors
        if counter>=3:
            counter = counter-1
        if counter>=5:
            counter = counter-2
        return counter
    
    def count_openings (self, img):
        # Function that counts the number of openings in a house
        # most of the openings are quardrilaterals which denote windows and gates
        
        # convert to grayscale
        gray_real = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # bilateral filtering
        img_filtered = cv2.bilateralFilter(gray_real, self.bilateralFilter_diameter, self.bilateralFilter_sigmaColor, self.bilateralFilter_sigmaSpace)
        # ret, thresh_real = cv2.threshold(img_filtered, 75, 255, 0)
        # adaptive thresholding
        thresh_real = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptiveThreshold_blockSize, self.adaptiveThreshold_C)
        # contour detection
        contours, h = cv2.findContours(thresh_real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate polygons from contours
        approx_polygons = [cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) for contour in contours]
        # area of the image
        area = thresh_real.shape[0]*thresh_real.shape[1]
        # filter contours which have area>0.1% of image area
        polygons = [polygon for polygon in approx_polygons if cv2.contourArea(polygon) > 0.001*area]
        # get convex hulls of selected polygons 
        convex_polygons = [cv2.convexHull(polygon) for polygon in polygons if len(polygon)]
        # filter by number of sides (between 3 and 6) and area of polygons (between 0.1% and 15% of the image area)
        final_polygons = [cv2.convexHull(polygon) for polygon in convex_polygons if len(polygon) >= 3 and len(polygon) <= 6 
                          and cv2.contourArea(polygon) > 0.001*area and cv2.contourArea(polygon) < 0.15*area]    
        
        
        ## remove polygons which are too close to the borders
        redflag = [0]*len(final_polygons)
        for i in range(len(final_polygons)):
            q = final_polygons[i]
            for point in q:
                if abs(point[0][0] - gray_real.shape[1]) < 50 or abs(point[0][0]) < self.borderMargin:
                    redflag[i] = 1
                if abs(point[0][1] - gray_real.shape[0]) < 50 or abs(point[0][1]) < self.borderMargin:
                    redflag[i] = 1
        
        ## final openings
        openings = [final_polygons[i] for i in range(len(final_polygons)) if redflag[i] != 1]
        return len(openings)
    
    def fraction_area(self,img):
        # Function that calculates fraction of sum of all openings' areas to the overall area of image
        
        # convert to grayscale
        gray_real = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # bilateral filtering
        img_filtered = cv2.bilateralFilter(gray_real, self.bilateralFilter_diameter, self.bilateralFilter_sigmaColor, self.bilateralFilter_sigmaSpace)
        # ret, thresh_real = cv2.threshold(img_filtered, 75, 255, 0)
        # adaptive thresholding
        thresh_real = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptiveThreshold_blockSize, self.adaptiveThreshold_C)
        # contour detection
        contours, h = cv2.findContours(thresh_real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate polygons from contours
        approx_polygons = [cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) for contour in contours]
        # area of the image
        area = thresh_real.shape[0]*thresh_real.shape[1]
        # filter contours which have area>0.1% of image area
        polygons = [polygon for polygon in approx_polygons if cv2.contourArea(polygon) > 0.001*area]
        # get convex hulls of selected polygons 
        convex_polygons = [cv2.convexHull(polygon) for polygon in polygons if len(polygon)]
        # filter by number of sides (between 3 and 6) and area of polygons (between 0.1% and 15% of the image area)
        final_polygons = [cv2.convexHull(polygon) for polygon in convex_polygons if len(polygon) >= 3 and len(polygon) <= 6 
                          and cv2.contourArea(polygon) > 0.001*area and cv2.contourArea(polygon) < 0.15*area]    
        
        
        ## remove polygons which are too close to the borders
        redflag = [0]*len(final_polygons)
        for i in range(len(final_polygons)):
            q = final_polygons[i]
            for point in q:
                if abs(point[0][0] - gray_real.shape[1]) < 50 or abs(point[0][0]) < self.borderMargin:
                    redflag[i] = 1
                if abs(point[0][1] - gray_real.shape[0]) < 50 or abs(point[0][1]) < self.borderMargin:
                    redflag[i] = 1
        
        ## final openings
        openings = [final_polygons[i] for i in range(len(final_polygons)) if redflag[i] != 1]
        final_polygon_areas = [cv2.contourArea(polygon) for polygon in openings]    
        
        return np.sum(final_polygon_areas)/area
    

    def fraction_width(self, img):
        # Function that calculates proportion of sum of all windows' widths (without overlap), on all floors 
        # to the overall wigth of building
        height = img.shape[0]
        width = img.shape[1]
        # convert to grayscale
        gray_real = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # bilateral filtering
        img_filtered = cv2.bilateralFilter(gray_real, self.bilateralFilter_diameter, self.bilateralFilter_sigmaColor, self.bilateralFilter_sigmaSpace)
        # ret, thresh_real = cv2.threshold(img_filtered, 75, 255, 0)
        # adaptive thresholding
        thresh_real = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptiveThreshold_blockSize, self.adaptiveThreshold_C)
        # contour detection
        contours, h = cv2.findContours(thresh_real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate polygons from contours
        approx_polygons = [cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) for contour in contours]
        # area of the image
        area = thresh_real.shape[0]*thresh_real.shape[1]
        # filter contours which have area>0.1% of image area
        polygons = [polygon for polygon in approx_polygons if cv2.contourArea(polygon) > 0.001*area]
        # get convex hulls of selected polygons 
        convex_polygons = [cv2.convexHull(polygon) for polygon in polygons if len(polygon)]
        # filter by number of sides (between 3 and 6) and area of polygons (between 0.1% and 15% of the image area)
        final_polygons = [cv2.convexHull(polygon) for polygon in convex_polygons if len(polygon) >= 3 and len(polygon) <= 6 
                          and cv2.contourArea(polygon) > 0.001*area and cv2.contourArea(polygon) < 0.15*area]    
        
        
        ## remove polygons which are too close to the borders
        redflag = [0]*len(final_polygons)
        for i in range(len(final_polygons)):
            q = final_polygons[i]
            for point in q:
                if abs(point[0][0] - gray_real.shape[1]) < 50 or abs(point[0][0]) < self.borderMargin:
                    redflag[i] = 1
                if abs(point[0][1] - gray_real.shape[0]) < 50 or abs(point[0][1]) < self.borderMargin:
                    redflag[i] = 1
        
        ## final openings
        openings = [final_polygons[i] for i in range(len(final_polygons)) if redflag[i] != 1]
        
        # Get a blank canvas for drawing width of a side of each quadrilateral
        detection_series = np.zeros(width, dtype = 'uint8')

        # The width of a side should be the larger x cordinate of the right vertics 
        # minus the x cordinate of the left vertics
        for i in range(len(openings)):
            q = openings[i]
            x_min = np.min(q[:,0,0])
            x_max = np.max(q[:,0,0])
            detection_series[x_min:x_max] = np.ones(x_max-x_min, dtype = 'uint8')
        # Return fraction of sum of all windows' widths (without overlap), on all floors to the overall width of building
        return np.sum(detection_series)/width
    
    def avg_fraction_width(self, img):
        # Function that calculates proportion of sum of all windows' widths (divided by the number of floors) 
        # to the overall length of building

        height = img.shape[0]
        width = img.shape[1]
        # convert to grayscale
        gray_real = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # bilateral filtering
        img_filtered = cv2.bilateralFilter(gray_real, self.bilateralFilter_diameter, self.bilateralFilter_sigmaColor, self.bilateralFilter_sigmaSpace)
        # ret, thresh_real = cv2.threshold(img_filtered, 75, 255, 0)
        # adaptive thresholding
        thresh_real = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptiveThreshold_blockSize, self.adaptiveThreshold_C)
        # contour detection
        contours, h = cv2.findContours(thresh_real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate polygons from contours
        approx_polygons = [cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) for contour in contours]
        # area of the image
        area = thresh_real.shape[0]*thresh_real.shape[1]
        # filter contours which have area>0.1% of image area
        polygons = [polygon for polygon in approx_polygons if cv2.contourArea(polygon) > 0.001*area]
        # get convex hulls of selected polygons 
        convex_polygons = [cv2.convexHull(polygon) for polygon in polygons if len(polygon)]
        # filter by number of sides (between 3 and 6) and area of polygons (between 0.1% and 15% of the image area)
        final_polygons = [cv2.convexHull(polygon) for polygon in convex_polygons if len(polygon) >= 3 and len(polygon) <= 6 
                          and cv2.contourArea(polygon) > 0.001*area and cv2.contourArea(polygon) < 0.15*area]    
        
        
        ## remove polygons which are too close to the borders
        redflag = [0]*len(final_polygons)
        for i in range(len(final_polygons)):
            q = final_polygons[i]
            for point in q:
                if abs(point[0][0] - gray_real.shape[1]) < 50 or abs(point[0][0]) < self.borderMargin:
                    redflag[i] = 1
                if abs(point[0][1] - gray_real.shape[0]) < 50 or abs(point[0][1]) < self.borderMargin:
                    redflag[i] = 1
        
        ## final openings
        openings = [final_polygons[i] for i in range(len(final_polygons)) if redflag[i] != 1]
        
        # set aggregate width = 0 before the loop that is going to account for the width of each opening
        aggregate_width = 0;

        # The width of a side should be the larger x cordinate of the right vertics 
        # minus the x cordinate of the left vertics
        for i in range(len(openings)):
            q = openings[i]
            x_min = np.min(q[:,0,0])
            x_max = np.max(q[:,0,0])
            aggregate_width = aggregate_width + (x_max-x_min)

        # now in order to calculate the average, we need the number of floors
        num_levels = self.count_level(img)

        # Return the ratio of: average of sum of all windows' widths (over all floors) to the total width of the building
        return aggregate_width/(num_levels*width)
    
    def fraction_height(self, img):
        # Function that calculates proportion of sum of all windows' heights (without overlap), on all floors
        # to the overall length of building
        
        height = img.shape[0]
        width = img.shape[1]
        # convert to grayscale
        gray_real = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # bilateral filtering
        img_filtered = cv2.bilateralFilter(gray_real, self.bilateralFilter_diameter, self.bilateralFilter_sigmaColor, self.bilateralFilter_sigmaSpace)
        # ret, thresh_real = cv2.threshold(img_filtered, 75, 255, 0)
        # adaptive thresholding
        thresh_real = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptiveThreshold_blockSize, self.adaptiveThreshold_C)
        # contour detection
        contours, h = cv2.findContours(thresh_real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate polygons from contours
        approx_polygons = [cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) for contour in contours]
        # area of the image
        area = thresh_real.shape[0]*thresh_real.shape[1]
        # filter contours which have area>0.1% of image area
        polygons = [polygon for polygon in approx_polygons if cv2.contourArea(polygon) > 0.001*area]
        # get convex hulls of selected polygons 
        convex_polygons = [cv2.convexHull(polygon) for polygon in polygons if len(polygon)]
        # filter by number of sides (between 3 and 6) and area of polygons (between 0.1% and 15% of the image area)
        final_polygons = [cv2.convexHull(polygon) for polygon in convex_polygons if len(polygon) >= 3 and len(polygon) <= 6 
                          and cv2.contourArea(polygon) > 0.001*area and cv2.contourArea(polygon) < 0.15*area]    
        
        
        ## remove polygons which are too close to the borders
        redflag = [0]*len(final_polygons)
        for i in range(len(final_polygons)):
            q = final_polygons[i]
            for point in q:
                if abs(point[0][0] - gray_real.shape[1]) < 50 or abs(point[0][0]) < self.borderMargin:
                    redflag[i] = 1
                if abs(point[0][1] - gray_real.shape[0]) < 50 or abs(point[0][1]) < self.borderMargin:
                    redflag[i] = 1
        
        ## final openings
        openings = [final_polygons[i] for i in range(len(final_polygons)) if redflag[i] != 1]
        
        # Get a blank canvas for drawing width of a side of each quadrilateral
        detection_series = np.zeros(height, dtype = 'uint8')

        # The height of a side should be the larger y cordinate of the top vertics 
        # minus the y cordinate of the bottom vertics
        for i in range(len(openings)):
            q = openings[i]
            if redflag[i]!=1:
                y_min = np.min(q[:,0,1])
                y_max = np.max(q[:,0,1])
                detection_series[y_min:y_max] = np.ones(y_max-y_min, dtype = 'uint8')

        # Return fraction of sum of all windows' heights (without overlap), on all floors to the overall length of building
        return np.sum(detection_series)/height
    
    def aggregate_fraction_height(self, img):
        # Function that calculates proportion of sum of all windows' heights (divided by the number of floors) 
        # to the overall length of building
        
        height = img.shape[0]
        width = img.shape[1]
        # convert to grayscale
        gray_real = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # bilateral filtering
        img_filtered = cv2.bilateralFilter(gray_real, self.bilateralFilter_diameter, self.bilateralFilter_sigmaColor, self.bilateralFilter_sigmaSpace)
        # ret, thresh_real = cv2.threshold(img_filtered, 75, 255, 0)
        # adaptive thresholding
        thresh_real = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptiveThreshold_blockSize, self.adaptiveThreshold_C)
        # contour detection
        contours, h = cv2.findContours(thresh_real, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # approximate polygons from contours
        approx_polygons = [cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) for contour in contours]
        # area of the image
        area = thresh_real.shape[0]*thresh_real.shape[1]
        # filter contours which have area>0.1% of image area
        polygons = [polygon for polygon in approx_polygons if cv2.contourArea(polygon) > 0.001*area]
        # get convex hulls of selected polygons 
        convex_polygons = [cv2.convexHull(polygon) for polygon in polygons if len(polygon)]
        # filter by number of sides (between 3 and 6) and area of polygons (between 0.1% and 15% of the image area)
        final_polygons = [cv2.convexHull(polygon) for polygon in convex_polygons if len(polygon) >= 3 and len(polygon) <= 6 
                          and cv2.contourArea(polygon) > 0.001*area and cv2.contourArea(polygon) < 0.15*area]    
        
        
        ## remove polygons which are too close to the borders
        redflag = [0]*len(final_polygons)
        for i in range(len(final_polygons)):
            q = final_polygons[i]
            for point in q:
                if abs(point[0][0] - gray_real.shape[1]) < 50 or abs(point[0][0]) < self.borderMargin:
                    redflag[i] = 1
                if abs(point[0][1] - gray_real.shape[0]) < 50 or abs(point[0][1]) < self.borderMargin:
                    redflag[i] = 1
        
        ## final openings
        openings = [final_polygons[i] for i in range(len(final_polygons)) if redflag[i] != 1]
        
        # set aggregate height = 0 before the loop that is going to account for the height of each opening
        aggregate_height = 0

        # The height of a side should be the larger y cordinate of the top vertics 
        # minus the y cordinate of the bottom vertics
        for i in range(len(openings)):
            q = openings[i]
            y_min = np.min(q[:,0,1])
            y_max = np.max(q[:,0,1])
            aggregate_height = aggregate_height + (y_max-y_min)

        # To be careful:
        # Width of each floor is same and equal to the width of the house
        # However, height of each floor = height of the house / 3
        # So we actually don't need the number of floors to calculate the average

        # there is no notion of vertical floors, so this ratio is going to exceed one
        # Return the ratio of: sum of all windows' height to the total height of the building
        return aggregate_height/height