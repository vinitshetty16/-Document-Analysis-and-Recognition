#!/usr/bin/env python
# coding: utf-8

# # SIT789 - Applications of Computer Vision and Speech Processing
# ## Distinction Task 2.3: Document analysis and recognition

# #### Task 1. Hough transform for document skew estimation

# ##### Step 1. Load image from file and binarise the image using a threshold, e.g., 200 (see Section 2.1 in Task 2.2C for reference).

# In[1]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import statistics
import math
import time

doc = cv.imread('E:\T2 2023\SIT789 CV\Assignments\Task 2.2C\doc.jpg', 0)


# In[2]:


threshold = 200
ret, doc_bin = cv.threshold(doc, threshold, 255, cv.THRESH_BINARY)
plt.imshow(doc_bin)


# ##### Step 2. Get negative version of the binarised image by subtracting the binarised image from 255

# In[3]:


negative_image = 255 - doc_bin #convert black/white to white/black


# In[4]:


plt.imshow(negative_image)


# ##### Step 3. Extract connected components from the negative image.

# In[5]:


num_labels, labels_im = cv.connectedComponents(negative_image)


# In[6]:


num_labels


# ##### Step 4. Select candidate points on the negative image using one of the following strategies

# In[7]:


candidate_points_list = []
for current_label in range(0, num_labels):
    # Extract points corresponding to the current label
    current_label_points = np.column_stack(np.where(labels_im == current_label))
    
    if len(current_label_points) > 0:
        center_of_points = np.mean(current_label_points, axis=0)
        
        # Convert center coordinates to integers
        center_points_int = center_of_points.astype(np.int32)
        
        candidate_points_list.append(center_points_int)


# ##### Step 5. Remove all pixels which are not candidate points from the negative image.

# In[8]:


new_negative_image = np.zeros_like(negative_image)

for point in candidate_points_list:
    y,x = point
    if y < new_negative_image.shape[0] and x < new_negative_image.shape[1]:
        new_negative_image[y,x] = 255


# In[9]:


plt.imshow(new_negative_image)
cv.imwrite('new_negative_image.png', new_negative_image)


# ##### Step 6. Define parameters for Hough space.

# In[10]:


spatial_res = 1
angular_res = np.pi/180
threshold = 10


# ##### Step 7. Apply Hough transform on the negative image

# In[11]:


lines = cv.HoughLines(new_negative_image, spatial_res, angular_res, threshold)


# ##### Step 8. Create an array to store the angles of all lines detected by Hough transform

# In[12]:


line_angles = []

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        
        # Convert theta from radians to degrees
        angle_degrees = theta * 180 / np.pi
        line_angles.append(angle_degrees)


# ##### Step 9. Import statistics. Then apply statistics.median to the array of angles created in Step 8. This will result in the median angle. However, this angle is NOT the document's angle. The document's angle is orthogonal to the median angle.

# In[13]:


median_angle = statistics.median(line_angles)

# Calculate the document's angle by adding or subtracting 90 degrees from the median angle
document_angle = median_angle - 90


# In[14]:


print(median_angle)
document_angle


# ##### Step 10. Deskew the image in doc.jpg with the angle calculated in Step 9

# In[15]:


# rotate image
height, width = doc.shape
c_x = (width - 1) / 2.0 # column index varies in [0, width-1]
c_y = (height - 1) / 2.0 # row index varies in [0, height-1]
c = (c_x, c_y) # A point is defined by x and y coordinate

M = cv.getRotationMatrix2D(c, document_angle, 1)
doc_deskewed_o = cv.warpAffine(doc, M, (width, height))

plt.imshow(doc_deskewed_o, 'gray')
cv.imwrite('doc_deskewed_o.png', doc_deskewed_o)


# #### Task 2. Performance analysis

# ##### Step 2.1. Candidate point selection

# In[16]:


def select_candidate_points(image, num_labels, labels_im , strategy, density_threshold):
    
    candidate_points_list = []
    for current_label in range(0, num_labels):
        # Extract points corresponding to the current label
        current_label_points = np.column_stack(np.where(labels_im == current_label))
        if len(current_label_points) > 0:
            
            # Strategy 'a': All foreground pixels are candidate points.
            if strategy == 'a':
                candidate_points_list += current_label_points.tolist()   
                
            # Strategy 'b': Centers of connected components are candidate points.
            elif strategy == 'b':
                center_of_points = np.mean(current_label_points, axis=0)
                
                # Convert center coordinates to integers
                center_points_int = center_of_points.astype(np.int32)
                
                candidate_points_list.append(center_points_int)
                
            # Strategy 'c': Points with maximum y-coordinate in each connected component are candidate points.   
            elif strategy == 'c':
                
                max_y = np.argmax(np.max(current_label_points, axis=1))
                max_y_points = current_label_points[max_y]
                candidate_points_list.append(max_y_points)


    return candidate_points_list


# ##### Step 2.2 Parameter setting

# In[17]:


def estimate_skew_angle(image_path, candidate_point_strategy, density_threshold):
    
    start_time = time.time()
    
    # Step 1: Load image from file and binarise
    doc = image_path
    ret, doc_bin = cv.threshold(doc, 200, 255, cv.THRESH_BINARY)
    
    
    # Step 2: Get negative version of the binarised image
    negative_image = 255 - doc_bin
    
    #Step 3. Extract connected components from the negative image.
    num_labels, labels_im = cv.connectedComponents(negative_image)
    
    # Step 4: Select candidate points based on the provided strategy
    candidate_points_list = select_candidate_points(negative_image, num_labels, labels_im, candidate_point_strategy, density_threshold)  
     
        
    # Step 5. Remove all pixels which are not candidate points from the negative image
    new_negative_image = np.zeros_like(negative_image)
    
    for point in candidate_points_list:
        y,x = point
        #To check if a point lies within the bounds of an image
        if 0 <= y < new_negative_image.shape[0] and 0 <= x < new_negative_image.shape[1]:
            new_negative_image[y,x] = 255
            
    # Step 6: Define parameters for Hough space
    spatial_res = 1
    angular_res = np.pi/180
    
    
    # Step 7: Apply Hough transform on the negative image
    lines = cv.HoughLines(new_negative_image, spatial_res, angular_res, density_threshold)
    
    
    # Step 8: Create an array to store the angles of all lines detected by Hough transform
    line_angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Convert theta from radians to degrees
            angle_degrees = theta * 180 / np.pi
            line_angles.append(angle_degrees)
        
    # Step 9: Calculate the median angle and find the document's angle (orthogonal)
    median_angle = statistics.median(line_angles)
    
    # Calculate the document's angle by adding or subtracting 90 degrees from the median angle
    
    document_angle = median_angle - 90
    
    
    # Step 10: rotate image
    height, width = doc.shape
    c_x = (width - 1) / 2.0 # column index varies in [0, width-1]
    c_y = (height - 1) / 2.0 # row index varies in [0, height-1]
    c = (c_x, c_y) # A point is defined by x and y coordinate
    
    M = cv.getRotationMatrix2D(c, document_angle, 1)
    doc_deskewed = cv.warpAffine(doc, M, (width, height))
    
    end_time = time.time()
    computational_time = end_time - start_time
    
    plt.imshow(doc_deskewed, 'gray')
    cv.imwrite('doc_deskewed.png', doc_deskewed)
    
    return document_angle, computational_time, doc_deskewed


# In[18]:


strategies = ['a', 'b', 'c']
thresholds = [10, 15, 20]
doc = cv.imread('E:\T2 2023\SIT789 CV\Assignments\Task 2.2C\doc.jpg', 0)

for strategy in strategies:
    for threshold in thresholds:
        document_angle, computational_time, doc_deskewed = estimate_skew_angle(doc, strategy, threshold)
        print(f"Strategy: {strategy}, Threshold: {threshold}")
        print(f"Estimated skew angle: {document_angle:.2f} degrees")
        print(f"Skew estimation took {computational_time:.6f} seconds.")
        # Display the deskewed image using matplotlib
        plt.imshow(doc_deskewed, 'gray')
        plt.show()
        filename = f"deskewed_{strategy}_{threshold}.jpg"
        cv.imwrite(filename, doc_deskewed)
        print("-------------------")

print("Skew estimation for all strategies and thresholds completed.")


# **The best strategy was found to be 'b' i.e. The centers of connected components are considered as the candidate points with a density threshold of '20'. This strategy resulted in the most precise deskewed image, featuring a skew angle of -13 degrees, with a computational time of only 64.61 seconds.**

# #### Task 3.Other test cases

# In[19]:


doc_1 = cv.imread('E:\T2 2023\SIT789 CV\Assignments\Task 2.3D\doc_1.jpg', 0)
doc_2 = cv.imread('E:\T2 2023\SIT789 CV\Assignments\Task 2.3D\doc_2.jpg', 0)
doc_3 = cv.imread('E:\T2 2023\SIT789 CV\Assignments\Task 2.3D\doc_3.jpg', 0)
doc_4 = cv.imread('E:\T2 2023\SIT789 CV\Assignments\Task 2.3D\doc_4.jpg', 0)
doc_5 = cv.imread('E:\T2 2023\SIT789 CV\Assignments\Task 2.3D\doc_5.jpg', 0)


# ###### Skew Estimation for doc_1 Image

# In[20]:


candidate_point_strategy = 'c'
density_threshold = 15

document_angle, computational_time, doc_deskewed_1 = estimate_skew_angle(doc_1, candidate_point_strategy,  density_threshold)
print(f"Estimated skew angle: {document_angle:.2f} degrees")
print(f"Skew estimation took {computational_time:.6f} seconds.")

# Display the Original and deskewed image using matplotlib
plt.imshow(doc_1,'gray')
plt.title('Original Image')
cv.imwrite('Original_doc_1.png', doc_1)
plt.show()

plt.imshow(doc_deskewed_1, 'gray')
plt.title('Deskewed Image')
cv.imwrite('deskewed_doc_1.png', doc_deskewed_1)
plt.show()


# - For "doc_1" Image, the candidate point strategy used was "The point which has maximum y-coordinate in each connected component" with density threshold of "15".
# - This strategy resulted in the most precise deskewed image, featuring a skew angle of 77 degrees, and with a computational time of 70.72 seconds.

# ###### Skew Estimation for doc_2 Image

# In[21]:


candidate_point_strategy = 'c'
density_threshold = 10

document_angle, computational_time, doc_deskewed_2 = estimate_skew_angle(doc_2, candidate_point_strategy,  density_threshold)
print(f"Estimated skew angle: {document_angle:.2f} degrees")
print(f"Skew estimation took {computational_time:.6f} seconds.")

# Display the Original and deskewed image using matplotlib
plt.imshow(doc_2,'gray')
plt.title('Original Image')
plt.show()
cv.imwrite('Original_doc_2.png', doc_2)

plt.imshow(doc_deskewed_2, 'gray')
plt.title('Deskewed Image')
cv.imwrite('deskewed_doc_2.png', doc_deskewed_2)
plt.show()


# - For "doc_2" Image, the candidate point strategy used was "The point which has maximum y-coordinate in each connected component" with density threshold of "10".
# - This strategy resulted in the most precise deskewed image, featuring a skew angle of 4 degrees, and with a computational time of 0.294 seconds.

# ###### Skew Estimation for doc_3 Image

# In[22]:


candidate_point_strategy = 'c'
density_threshold = 10

document_angle, computational_time, doc_deskewed_3 = estimate_skew_angle(doc_3, candidate_point_strategy,  density_threshold)
print(f"Estimated skew angle: {document_angle:.2f} degrees")
print(f"Skew estimation took {computational_time:.6f} seconds.")

# Display the Original and deskewed image using matplotlib
plt.imshow(doc_3,'gray')
plt.title('Original Image')
cv.imwrite('Original_doc_3.png', doc_3)
plt.show()

plt.imshow(doc_deskewed_3, 'gray')
plt.title('Deskewed Image')
cv.imwrite('deskewed_doc_3.png', doc_deskewed_3)
plt.show()


# - For "doc_3" Image, the candidate point strategy used was "The point which has maximum y-coordinate in each connected component" with density threshold of "10".
# - This strategy resulted in the most precise deskewed image, featuring a skew angle of 15 degrees, and with a computational time of 1.76 seconds.

# ###### Skew Estimation for doc_4 Image

# In[23]:


candidate_point_strategy = 'c'
density_threshold = 10

document_angle, computational_time, doc_deskewed_4 = estimate_skew_angle(doc_4, candidate_point_strategy,  density_threshold)
print(f"Estimated skew angle: {document_angle:.2f} degrees")
print(f"Skew estimation took {computational_time:.6f} seconds.")

# Display the Original and deskewed image using matplotlib
plt.imshow(doc_4,'gray')
plt.title('Original Image')
cv.imwrite('Original_doc_4.png', doc_4)
plt.show()

plt.imshow(doc_deskewed_4, 'gray')
plt.title('Deskewed Image')
cv.imwrite('deskewed_doc_4.png', doc_deskewed_4)
plt.show()


# - For "doc_4" Image, the candidate point strategy used was "The point which has maximum y-coordinate in each connected component" with density threshold of "10".
# - This strategy resulted in the most precise deskewed image, featuring a skew angle of -2 degrees, and with a computational time of 1.74 seconds.

# ###### Skew Estimation for doc_5 Image

# In[24]:


candidate_point_strategy = 'c'
density_threshold = 10

document_angle, computational_time, doc_deskewed_5 = estimate_skew_angle(doc_5, candidate_point_strategy,  density_threshold)
print(f"Estimated skew angle: {document_angle:.2f} degrees")
print(f"Skew estimation took {computational_time:.6f} seconds.")

# Display the Original and deskewed image using matplotlib
plt.imshow(doc_5,'gray')
plt.title('Original Image')
cv.imwrite('Original_doc_5.png', doc_5)
plt.show()

plt.imshow(doc_deskewed_5, 'gray')
plt.title('Deskewed Image')
cv.imwrite('deskewed_doc_5.png', doc_deskewed_5)
plt.show()


# - For "doc_5" Image, the candidate point strategy used was "The point which has maximum y-coordinate in each connected component" with density threshold of "10".
# - This strategy resulted in the most precise deskewed image, featuring a skew angle of 5 degrees, and with a computational time of 1.21 seconds.

# #### 4. Text recognition using pytesseract

# In[26]:


import pytesseract

# Set the Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[27]:


doc = cv.imread('E:\T2 2023\SIT789 CV\Assignments\Task 2.2C\doc.jpg', 0)

text = pytesseract.image_to_string(doc)


# In[28]:


print(text)


# In[29]:


# Generate PDF data using Tesseract OCR

doc_pdf = pytesseract.image_to_pdf_or_hocr(doc, extension='pdf')

# Save the PDF data to a file named 'doc_pdf'
with open('doc.pdf', 'w+b') as f:
    f.write(doc_pdf)

print("PDF saved successfully.")


# ##### Text recognition on the deskewed doc

# In[30]:


doc_deskewed = cv.imread('C:/Users/vinit/doc_deskewed_o.png', 0)

Deskewed_text = pytesseract.image_to_string(doc_deskewed)


# In[31]:


print(Deskewed_text)


# In[32]:


# Generate PDF data using Tesseract OCR
deskewed_doc_data = pytesseract.image_to_pdf_or_hocr(doc_deskewed_o, extension='pdf')

# Save the PDF data to a file named 'img.pdf'
with open('doc_deskewed.pdf', 'w+b') as f:
    f.write(deskewed_doc_data)

print("PDF saved successfully.")


# ##### Text Recognition on Deskewed Image doc_deskewed_4.png

# In[33]:


doc_deskewed_4 = cv.imread('C:/Users/vinit/deskewed_doc_4.png', 0)

Deskewed_text_4 = pytesseract.image_to_string(doc_deskewed_4)


# In[34]:


print(Deskewed_text_4)


# In[35]:


# Generate PDF data using Tesseract OCR
deskewed_doc_data_4 = pytesseract.image_to_pdf_or_hocr(doc_deskewed_4, extension='pdf')

# Save the PDF data to a file named 'img.pdf'
with open('doc_deskewed_4.pdf', 'w+b') as f:
    f.write(deskewed_doc_data_4)

print("PDF saved successfully.")

