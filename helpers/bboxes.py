import csv
from torch import zeros_like
import numpy as np 

def resize_bboxes(bboxes, original_size, target_size=512):
    """
    Resize bounding boxes to match a new image size.

    Args:
    - bboxes (list of lists): List of bounding boxes [x, y, width, height]
    - original_size (int): Original size of the image (e.g., 224, 1024)
    - target_size (int): Target size of the image (default is 512)

    Returns:
    - List of resized bounding boxes
    """
    scale = target_size / original_size
    resized_bboxes = [[x * scale, y * scale, w * scale,
                       h * scale] for x, y, w, h in bboxes]
    return resized_bboxes


def read_bounding_boxes(bbox_csv_path):
    """
    Read the bounding box CSV file and return a dictionary with 
    image names as keys and bounding boxes as values.
    """
    bbox_dict = {}

    with open(bbox_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_name = row['img_name']
            # Convert string to a list
            # (e.g., "[32, 119, 15, 13]" -> [32, 119, 15, 13])
            bbox_224 = eval(row['bbox_224'])
            # Store the bounding box (x, y, width, height)
            bbox_dict[img_name] = bbox_224

    return bbox_dict

def calculate_cam_iou(cam_mask, bounding_boxes):
    """Calculate IoU between CAM mask and ground-truth bounding boxes."""
    # cam_mask = (cam_mask > 0.5).float()  # Binary mask
    ious = []
    #range(cam_mask.size(0)):
    for i in range(cam_mask.shape[0]):
        # Get ground-truth box (xmin, ymin, xmax, ymax)
        box = bounding_boxes[i]  
        cam_region = cam_mask[i].nonzero()  # Get CAM region
        
        if cam_region.size(0) == 0:  # If no region, IoU is 0
            ious.append(0.0)
            continue
        
        # Create a mask for the bounding box
        box_mask = zeros_like(cam_mask[i])
        box_mask[box[1]:box[3], box[0]:box[2]] = 1.0

        # Calculate intersection and union between CAM region and 
        # ground-truth box
        intersection = (cam_mask[i] * box_mask).sum()
        union = (cam_mask[i] + box_mask).clamp(0, 1).sum()

        iou = intersection / union
        ious.append(iou.item())

    return ious
#  ==========================================

# calculations

def get_iou(true_bbox, pred_bbox):
    intersect_x1 = np.max(true_bbox[0], pred_bbox[0])
    intersect_y1 = np.max(true_bbox[1], pred_bbox[1])
    intersect_x2 = np.min(true_bbox[3], pred_bbox[3])
    intersect_y2 = np.min(true_bbox[4], pred_bbox[4])
    intersect_height = np.max(intersect_y2-intersect_y1, np.array(0.))
    intersect_width = np.max(intersect_x2-intersect_x1, np.array(0.))
    intersect_area = intersect_height*intersect_width

    true_height = true_bbox[3] - true_bbox[1]
    true_width =  true_bbox[2] - true_bbox[1]
    pred_height = pred_bbox[3] - pred_bbox[1]
    pred_width =  pred_bbox[2] - pred_bbox[1]
    
    union_area = true_height*true_width + pred_height * pred_width - intersect_area
    iou = intersect_area / union_area
    