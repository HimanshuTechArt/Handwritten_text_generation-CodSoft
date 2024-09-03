"""
Sources:
    - Font size for OCR:
        https://nuance.custhelp.com/app/answers/detail/a_id/6346/~/recommended-height-%28in-pixels%29-of-characters-for-optimal-ocr
        http://www.cvisiontech.com/library/ocr/best-ocr/best-font-and-size.html
        https://stackoverflow.com/questions/5718299/font-size-what-can-i-be-sure-of
"""
import random
import os
import cv2
import numpy as np
# from numba import jit

from Preprocessing import Preprocessing
preprocessing= Preprocessing()


class Generation():

    def __init__(self):
        """
        Attributes:
        - source_dir: Directory where to fetch handwritten characters images
        - charW: Specify character image width in order to resize it.
        - charH: Specify character image height in order to resize it.
        - char_capital_W: Specify capital character image width in order to resize it.
        - char_capital_H: Specify capital character image height in order to resize it.
        - keyword: keyword to look for among the filenames in source_dir.
        - dilation_kernel: kernel value, used for dilating images
        - separators: characters that can not be fetched in source_dir
        - special_char_dir: path in which images of special characters can be found 
                            (examples: ".", "/", ...)
        """
        self.source_dir= ""
        self.charW= 15
        self.charH= 25
        self.char_capital_W= 20
        self.char_capital_H= 30
        self.keyword= ""
        self.dilation_kernel = np.ones((2,2),np.uint8)
        self.separators= ["\n", "\t"]
        self.special_char_dir= "Data/HTR/special_characters/"

#    @jit(nopython=True)
    def generate_char_image(self, inputChar, remove_borders= True, resize= False, dilate= True):
        
        if inputChar != " ":
            if inputChar.isdigit():
              #in case inputChar is an integer
                int_number= int(inputChar)
                if int_number+1<10:
                    char_dir= self.source_dir+ f"Sample00{int_number+1}/"
                else:
                    char_dir= self.source_dir+ f"Sample0{int_number+1}/"
        
            elif inputChar == "/":
                char_dir= self.special_char_dir

            else:
                if inputChar.isupper():
                    int_number= ord(inputChar) - 54
                else:
                    int_number= ord(inputChar) - 60
#                     print(f"int_number: {int_number}")
                char_dir= self.source_dir+ f"Sample0{int_number}/"


            char_images= [x for x in os.listdir(char_dir) if self.keyword in x]
            image= cv2.imread(char_dir+ random.choice(char_images))
            
            if remove_borders:
                temp_img= image.copy()
                _, cntrs= preprocessing.draw_bounding_box_text(temp_img, return_contours=True)
                x, y, w, h = cv2.boundingRect(cntrs[-1])
                image= image[y:y+h, x:x+w]
        
            if resize:
                if not(inputChar.isdigit()) and inputChar.islower():
                    image= preprocessing.resize_image(image, new_width= self.charW, new_height= self.charH)
                    image= preprocessing.add_border(image, bordersize=0, top_size= self.char_capital_H - self.charH)
                else:
                    image= preprocessing.resize_image(image, new_width= self.char_capital_W, new_height= self.char_capital_H)
    
            if dilate:
                # image= preprocessing.image_threshold(image)
                image = cv2.dilate(image, self.dilation_kernel ,iterations = 1)
        else:
            image = 255 * np.ones(shape=[self.char_capital_H, self.char_capital_W, 3], dtype=np.uint8)
        
        return image

    def generate_string_image(self, inputStr, dilate= False):
        string_image= self.generate_char_image(inputStr[0], remove_borders=True, resize=True)
    
        for char in inputStr[1:]:
          image= self.generate_char_image(char, remove_borders=True, resize=True)
          string_image= (cv2.hconcat([string_image, image]))
    
        if dilate:
            string_image = cv2.dilate(string_image, self.dilation_kernel ,iterations = 1)
            
        return string_image
        
#     @jit(nopython=True)
    def generate_sentence_image(self, inputParagraph):
    
      sentences= (inputParagraph.replace("\t", "   ")).split("\n")
      
      sentence_img= self.generate_string_image(sentences[0])
      img_dict= dict()
      img_dict[0]= sentence_img
      img_shapes= [np.shape(sentence_img)]
      max_w= np.shape(sentence_img)[1]
    
      for i in range(1, len(sentences)):
        sentence_img= self.generate_string_image(sentences[i])
        img_dict[i]= sentence_img
        img_shapes.append(np.shape(sentence_img))
        if max_w < np.shape(sentence_img)[1]:
          max_w = np.shape(sentence_img)[1]
    
      result_img= preprocessing.add_border(img_dict[0], bordersize=0, right_size= max_w- img_shapes[0][1])
      for i in range(1, len(sentences)):
          temp_img= preprocessing.add_border(img_dict[i], bordersize=0, right_size= max_w- img_shapes[i][1])
          result_img= cv2.vconcat([result_img, temp_img])
    
      return result_img