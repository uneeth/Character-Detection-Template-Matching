"""
Character Detection

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def elementwise_mul_mean_sum(mat1, mat2, mean1, mean2):
    """Elementwise multiplication."""
    val = 0.0
    for k, row in enumerate(mat1):
        for l, num in enumerate(row):
            val += (mat1[k][l] - mean1)*(mat2[k][l] - mean2)
    return val

def elementwise_square_sum(mat, mean) :
    val = 0.0
    for k, row in enumerate(mat):
        for l, num in enumerate(row):
            val += (mat[k][l]- mean)**2
    return val

def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    coordinates=[]
    img=np.asarray(img)
    img=normalize(img)
    #img = detect_edges(img, prewitt_x, False)
    template = np.asarray(template)
    template=normalize(template)
    #img=rgb2gray(img)
    x = -1
    y = -1
    ncc=0
    max_value = -1
    template_rows = int(len(template))
    template_columns = int(len(template[0]))
    image_rows = len(img)
    image_columns = len(img[0])
    for i, row in enumerate(img):
        for j, num in enumerate(row):
            
            if i + template_rows > image_rows or j + template_columns > image_columns: 
                continue
            #print(i,j)
            cropped_img = utils.crop(img,i,i+ template_rows , j , j+template_columns)
            patch_value = 0.0
            increment = 0
            for k, row in enumerate(cropped_img):
                for l, column in enumerate(row):
                    patch_value += cropped_img[k][l]
                    increment += 1
            patch_mean = (patch_value*1.0)/increment
            
            template_value = 0.0
            additive = 0
            for a, r in enumerate(template):
                for b, c in enumerate(r):
                    template_value += template[a][b]
                    additive += 1
            template_mean = (template_value*1.0)/additive
            
            elementwise_mul_mean_sum_value = elementwise_mul_mean_sum(cropped_img, template, patch_mean, template_mean)
            elementwise_square_sum_patch_value = elementwise_square_sum(cropped_img, patch_mean)
            elementwise_square_sum_template_value = elementwise_square_sum(template, template_mean)
            if(elementwise_square_sum_patch_value!=0 and elementwise_square_sum_patch_value!=0):
                ncc = elementwise_mul_mean_sum_value/np.sqrt(elementwise_square_sum_patch_value * elementwise_square_sum_template_value)
            if(ncc>=0.9 and ncc<=1):        
                coordinates.append((j,i))
    #print(coordinates)
    # TODO: implement this function.
    # raise NotImplementedError
    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()