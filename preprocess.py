import torch
from IPython.display import Image  # for displaying images
import os
from pathlib import Path
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import yaml
from train import train

random.seed(108)


def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    # Initialise the info dict
    info_dict = {}
    info_dict["bboxes"] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name
        if elem.tag == "filename":
            info_dict["filename"] = elem.text

        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            info_dict["image_size"] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict["bboxes"].append(bbox)

    return info_dict


def extract_unique_class_names(xml_directory):
    class_names = set()
    for filename in os.listdir(xml_directory):
        if filename.endswith(".xml"):
            xml_file = os.path.join(xml_directory, filename)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall(".//object/name"):
                class_names.add(obj.text)
    return class_names


def convert_to_yolov8(info_dict, class_name_to_id_mapping, base_name):
    print_buffer = []
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width = b["xmax"] - b["xmin"]
        b_height = b["ymax"] - b["ymin"]

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(
                class_id, b_center_x, b_center_y, b_width, b_height
            )
        )

    # Name of the file which we have to save

    # base_name, ext = os.path.splitext(info_dict["filename"])
    save_file_name = os.path.join("converted_train", base_name + ".txt")

    file = open(save_file_name, "w")
    file.writelines(print_buffer)
    file.close()


def convert_train(data_path):
    # Get the annotations
    print("data_path", data_path)
    directory_to_delete = "converted_train"
    if os.path.exists(directory_to_delete) and os.path.isdir(directory_to_delete):
        try:
            shutil.rmtree(directory_to_delete)  # Removes the directory and its contents
            print(f"Directory '{directory_to_delete}' has been deleted.")
        except OSError as e:
            print(f"Error: {e}")
    else:
        print(f"Directory '{directory_to_delete}' does not exist.")

    annotations = [
        os.path.join(data_path, x) for x in os.listdir(data_path) if x[-3:] == "xml"
    ]
    annotations.sort()
    class_names = extract_unique_class_names(data_path)
    class_name_to_id_mapping = {
        class_name: class_id for class_id, class_name in enumerate(class_names)
    }

    # Convert and save the annotations
    # print(annotations)
    for ann in annotations:
        # print(ann)
        imgpath = ann[: len(ann) - 3]
        dst = imgpath[imgpath.find("\\") :]
        dst = "converted_train" + dst

        # Create the destination folder if it doesn't exist
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if Path(imgpath + "jpg").exists():
            shutil.copyfile(imgpath + "jpg", dst + "jpg")
        elif Path(imgpath + "png").exists():
            shutil.copyfile(imgpath + "png", dst + "png")
        elif Path(imgpath + "jpeg").exists():
            shutil.copyfile(imgpath + "jpeg", dst + "jpeg")
        info_dict = extract_info_from_xml(ann)

        basename = ann[ann.find("\\") + 1 : len(ann) - 4]
        # print(basename)
        convert_to_yolov8(info_dict, class_name_to_id_mapping, basename)


def split_data(data_directory, train_ratio=0.8, test_ratio=0.1, valid_ratio=0.1):
    image_files = [
        file for file in os.listdir(data_directory) if not file.endswith(".txt")
    ]
    text_files = [file for file in os.listdir(data_directory) if file.endswith(".txt")]

    # Ensure the corresponding image and text files match
    image_files.sort()
    text_files.sort()

    # Perform the train-test-valid split
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_files, text_files, test_size=1 - train_ratio, random_state=42
    )

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio / (test_ratio + valid_ratio),
        random_state=42,
    )

    # Create the train, test, and valid directories
    train_dir = os.path.join(data_directory, "train")
    test_dir = os.path.join(data_directory, "test")
    valid_dir = os.path.join(data_directory, "valid")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Create subdirectories for images and labels in train, test, and valid
    for subdir in ["images", "labels"]:
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, subdir), exist_ok=True)

    # Move the files to their respective directories
    for image_file, text_file in zip(X_train, y_train):
        shutil.move(
            os.path.join(data_directory, image_file),
            os.path.join(train_dir, "images", image_file),
        )
        shutil.move(
            os.path.join(data_directory, text_file),
            os.path.join(train_dir, "labels", text_file),
        )

    for image_file, text_file in zip(X_test, y_test):
        shutil.move(
            os.path.join(data_directory, image_file),
            os.path.join(test_dir, "images", image_file),
        )
        shutil.move(
            os.path.join(data_directory, text_file),
            os.path.join(test_dir, "labels", text_file),
        )

    for image_file, text_file in zip(X_valid, y_valid):
        shutil.move(
            os.path.join(data_directory, image_file),
            os.path.join(valid_dir, "images", image_file),
        )
        shutil.move(
            os.path.join(data_directory, text_file),
            os.path.join(valid_dir, "labels", text_file),
        )

    print(
        "Data split into train, test, and valid sets with separate subdirectories for images and labels."
    )

    class_names = extract_unique_class_names("train")
    print(class_names)
    yaml_data = {
        "train": "./converted_train/train/images",
        "val": "./converted_train/valid/images",
        "test": "./converted_train/test/images",
        "nc": len(class_names),
        "names": list(class_names),
    }
    with open("data.yaml", "w") as data_file:
        yaml.dump(yaml_data, data_file, default_flow_style=True)


# parameters: train_path, output model name
def start(data_path):
    convert_train(data_path)
    split_data("converted_train")
    train()


start("train")
