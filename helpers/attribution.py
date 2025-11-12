import cv2
from helpers.explainability import get_occ_int_grad_for_single_tensor
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import DBSCAN

def convert_to_uint8(image):
    """Helper function to convert images to cv2-compatible uint8 format. """
    uint8_image = image-image.min()
    uint8_image = uint8_image / uint8_image.max() * 255
    uint8_image = uint8_image.astype(np.uint8)
    return uint8_image

def get_attrs(image, label, model, int_model=None, occ_model=None, device="cpu", single=False, window=8):
    """
    Get occlusion and integrated gradient attributions for a given image.

    Args: 
        image (torch.float Tensor): The input image
        label (int): Index of image label (0: "Normal", 1: "Nodule").
        model (torchvision model): Inference model to obtain attributions from. 
        int_model (optional, Captum IntegratedGradients model): Int Grad model based on inference model.
        occ_model (optional, Captum Occlusion model): Occlusion model based on inference model.
        device (torch.device): Device to perform caluclations on (CUDA/CPU).
        single (boolean): Indicate whether image and model are single channel.
        window (int): Window size for occlusion window.

    Returns:
        int_attr: Integrated gradient attribution
        occ_attr: Occlusion attribution
    """
    
    model=model.to(device)
    int_attrs = get_occ_int_grad_for_single_tensor(int_model, image, label, single=single, window=window)
    occ_attrs = get_occ_int_grad_for_single_tensor(occ_model, image, label, single=single, window=window)
    model = model.to("cpu")
    cv2_occ_attr = cv2.cvtColor(occ_attrs[0][0].unsqueeze(2).cpu().numpy(), cv2.COLOR_RGB2BGR)
    cv2_int_attr = int_attrs[0][0].unsqueeze(2).cpu().numpy()
    return cv2_int_attr[0][0].unsqueeze(2).cpu().numpy(), cv2_occ_attr[0][0].unsqueeze(2).cpu().numpy()
    

def threshold_attributions(attr, min_thresh_frac=0.15, non_zero=False, binary_mask=False):
    if binary_mask:
        mask_type = cv2.THRESH_BINARY
    else:
        mask_type = cv2.THRESH_TOZERO
    
    thresh_attr = cv2.threshold(src = attr,
                                thresh = 0 if non_zero else attr.max()*min_thresh_frac, 
                                maxval = attr.max(),
                                type = mask_type)[1]
    return thresh_attr

def get_area_thresheld_connected_components(mask, size_thresh = 1, img=None):
    """
    Returns connected components from mask.

    Args:
        mask (): Single channel attributtion mask
        size_thresh (int, optional): minimum area for connected component

    Returns:
        df (DataFrame): ["centroid","xmin","xmax","ymin","ymax"]
        centroids (2D Numpy array): array of [x,y] pairs
    
    """
    df = pd.DataFrame(columns=["centroid","xmin","xmax","ymin","ymax"])
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convert_to_uint8(mask))
    if img is not None:
        copy_img = convert_to_uint8(img.copy())
    # save idx of valid components
    size_thresh_idx = []
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            size_thresh_idx.append(i)
            # Add a new row using loc[]
            df.loc[len(df)] = [centroids[i], x, x+w, y-h, y]
            if img is not None:
                cv2.rectangle(copy_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
                cv2.circle(copy_img, centroids[i].astype(int), 1, (255,0,0), thickness=1)
    if img is not None:
        plt.imshow(copy_img)
        plt.axis("off")
        
    return df


def attribution_bbox_with_connected_components(mask, size_thresh = 1, img=None):
    """
    Returns connected components from mask.

    Args:
        mask (array): Single channel attributtion mask
        size_thresh (int, optional): minimum area for connected component

    Returns:
        pandas.DataFrame: Connected components with cols ["centroid","xmin","xmax","ymin","ymax"].
        2D Numpy array: centroids, array of [x,y] pairs
    
    """
    df = pd.DataFrame(columns=["centroid","xmin","xmax","ymin","ymax"])
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convert_to_uint8(mask))
    if img is not None:
        copy_img = convert_to_uint8(img.copy())
    
    # save idx of valid components
    size_thresh_idx = []
    
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            size_thresh_idx.append(i)
            # Add a new row using loc[]
            df.loc[len(df)] = [centroids[i], x, x+w, y, y+h]
            if img is not None:
                cv2.rectangle(copy_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
                cv2.circle(copy_img, centroids[i].astype(int), 1, (255,0,0), thickness=1)
    if img is not None:
        plt.imshow(copy_img)
        plt.axis("off")
        
    return df
        
    
def plt_area_thresheld_connected_components(mask, plot_image, size_thresh = 1):
    """
    Visualise the connected componenents and centroids
    """
    # duplicate image to prevent overwriting base
    copy_img = convert_to_uint8(plot_image.copy())
    # generate cc
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convert_to_uint8(mask))
    
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # plot cc bbox
            cv2.rectangle(copy_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
            # plot cc centroid
            cv2.circle(copy_img, centroids[i].astype(int), 1, (255,0,0), thickness=1)
    plt.imshow(copy_img)
    plt.axis("off")


def get_area_thresheld_connected_components(mask, size_thresh = 1, img=None):
    """
    Returns connected components from mask.

    Args:
        mask (array): Single channel attributtion mask
        size_thresh (int, optional): minimum area for connected component

    Returns:
        df (DataFrame): ["centroid","xmin","xmax","ymin","ymax"]
        centroids (2D Numpy array): array of [x,y] pairs
    
    """
    df = pd.DataFrame(columns=["centroid","xmin","xmax","ymin","ymax"])
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convert_to_uint8(mask))
    if img is not None:
        copy_img = convert_to_uint8(img.copy())
    # save idx of valid components
    size_thresh_idx = []
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_thresh:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            size_thresh_idx.append(i)
            # Add a new row using loc[]
            df.loc[len(df)] = [centroids[i], x, x+w, y, y+h]
            if img is not None:
                cv2.rectangle(copy_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
                cv2.circle(copy_img, centroids[i].astype(int), 1, (255,0,0), thickness=1)
    if img is not None:
        plt.imshow(copy_img)
        plt.axis("off")
        
    return df
    # return centroids[size_thresh_idx].astype(int)

def df_cc_bboxes_by_dbscan(df, max_distance=10, min_samples=1, margin=0, x_shape=512, y_shape=512, thresh_area = 4):
    """
    Generate groupings of adjacent bounding boxes using DBSCAN
    """
    # get centroids from df and convert into correct format
    centroids = np.array(df['centroid'].values.tolist())
    # apply DBSCAN to centroids
    db = DBSCAN(eps=max_distance, min_samples=min_samples).fit(centroids)
    # get group labels from DBSCAN
    labels = db.labels_
    
    # Calculate bounding boxes for each cluster
    grouped_bboxes = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # cluster_points = centroids[labels == label]
        cluster_df = df[labels==label]
        
        # Calculate the bounding box for the cluster
        # x_min = np.min(cluster_points[:, 0])
        # y_min = np.min(cluster_points[:, 1])
        # x_max = np.max(cluster_points[:, 0])
        # y_max = np.max(cluster_points[:, 1])

        x_min = cluster_df.xmin.min()
        x_max = cluster_df.xmax.max()
        y_min = cluster_df.ymin.min()
        y_max = cluster_df.ymax.max()
        
        # Apply margin to expand the bounding box
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(x_shape - 1, x_max + margin)
        y_max = min(y_shape - 1, y_max + margin)

        if x_max-x_min > 0 and y_max-y_min > 0 and ((x_max-x_min)*(y_max-y_min)>=thresh_area):
            grouped_bboxes.append((x_min, y_min, x_max, y_max))
    
    return grouped_bboxes
    
def load_attr(pth):
    """Helper function to load exported attribute masks.
    
    Args:
        pth (string): path to attribution file

    Returns:
        attr: attribute mask array 
    """
    attr_mask = cv2.imread(pth)
    return attr_mask[:,:,0]

def get_attr_pth(attr_type, training_type, model_type, model_number, image_name, base_dir="/home/local/data/sophie/DGX_GridSearch/attributions"):
    """
    Craft path to exported attribute. Assumes path structure of :
    `[base_dir]/[attr_type]/[model_type]/[model_number]/[image_name]`.
    
    e.g.:
    `data/occlusion/full/base/model1/n00023.png`


    Args:
        attr_type: Expects 'occ' or 'int'
        training_type: Expects 'full', 'first' or 'half'
        model_type: Expects 'base','grey' or 'single'
        model_number: Expects seed (42,43,44) or run number (1,2,3)
        image_name: Expects nodule image name (e.g. 'n0023.png')
        
    """
    # add attr type to path
    if "occ" in attr_type.lower():
        base_dir = os.path.join(base_dir, "occlusion")
    elif "int" in attr_type.lower():
        base_dir = os.path.join(base_dir, "intgrad")
    else:
        raise ValueError("Please specify whether intgrad or occlusion is being loaded")
    # add training type to path
    if "full" in training_type.lower():
        base_dir = os.path.join(base_dir, "full")
    elif "first" in training_type.lower():
        base_dir = os.path.join(base_dir, "first")
    elif "half" in training_type.lower():
        base_dir = os.path.join(base_dir, "half")
    else:
        raise ValueError("Please specify the training strategy.")

    # add model type to path
    if "base65" in model_type.lower():
        base_dir = os.path.join(base_dir, "base65")
    elif "base" in model_type.lower():
        base_dir = os.path.join(base_dir, "base")
    elif "grey65" in model_type.lower():
        base_dir = os.path.join(base_dir, "grey65")
    elif "grey" in model_type.lower():
        base_dir = os.path.join(base_dir, "grey")
    elif "single65" in model_type.lower():
        base_dir = os.path.join(base_dir, "single65")
    elif "single" in model_type.lower():
        base_dir = os.path.join(base_dir, "single")
    else:
        raise ValueError("Please specify whether model is of type base, grey, or single")
    
    # add model number to path
    model_number = int(model_number)
    if model_number == 42 or model_number == 1:
        base_dir = os.path.join(base_dir, "model1")
    elif model_number == 43 or model_number == 2:
        base_dir = os.path.join(base_dir, "model2")
    elif model_number == 44 or model_number == 3:
        base_dir = os.path.join(base_dir, "model3")
    else:
        raise ValueError("Please specify either model seed (42, 43, 44) or model run number (1, 2, 3)")

    # add image name to path
    if "n" in image_name.lower():
        if ".png" not in image_name.lower():
            image_name += ".png"
        base_dir = os.path.join(base_dir, image_name)
    else:
        raise ValueError("Method only supports loading nodule images")
    
    return base_dir

    