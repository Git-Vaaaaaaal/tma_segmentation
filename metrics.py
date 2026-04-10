import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion as nd_binary_erosion

def dice_coefficient(y_true, y_pred, epsilon=1e-7):
    y_pred = (y_pred > 0.5).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + epsilon) / (np.sum(y_true) + np.sum(y_pred) + epsilon)

def surface_dice(y_true, y_pred, tolerance_mm=5.0, voxel_spacing=(1.0, 1.0)):
    true_surface = np.logical_xor(y_true, nd_binary_erosion(y_true))
    pred_surface = np.logical_xor(y_pred, nd_binary_erosion(y_pred))

    true_dist = distance_transform_edt(~true_surface, sampling=voxel_spacing)
    pred_dist = distance_transform_edt(~pred_surface, sampling=voxel_spacing)

    true_close = pred_dist < tolerance_mm
    pred_close = true_dist < tolerance_mm

    numerator = np.sum(true_surface & true_close) + np.sum(pred_surface & pred_close)
    denominator = np.sum(true_surface) + np.sum(pred_surface)

    return numerator / (denominator + 1e-7)

def average_symmetric_surface_distance(y_true, y_pred, voxel_spacing=(1.0, 1.0)):
    """Average Symmetric Surface Distance"""
    true_surface = np.logical_xor(y_true, nd_binary_erosion(y_true))
    pred_surface = np.logical_xor(y_pred, nd_binary_erosion(y_pred))

    true_dist = distance_transform_edt(~y_true, sampling=voxel_spacing)
    pred_dist = distance_transform_edt(~y_pred, sampling=voxel_spacing)

    assd_1 = np.mean(true_dist[pred_surface])
    assd_2 = np.mean(pred_dist[true_surface])
    return (assd_1 + assd_2) / 2.0



def compute_segmentation_metrics(y_trues, y_preds, tolerance_mm=5.0):
    results = []

    for i in range(len(y_trues)):
        y_true = (y_trues[i].squeeze() > 0.5).astype(np.uint8)
        y_pred = (y_preds[i].squeeze() > 0.5).astype(np.uint8)

        dice = dice_coefficient(y_true, y_pred)
        sdice = surface_dice(y_true, y_pred, tolerance_mm)
        assd = average_symmetric_surface_distance(y_true, y_pred)


        results.append({
            "Dice": dice,
            "SurfaceDice@5mm": sdice,
            "ASSD": assd,
        })

    df = pd.DataFrame(results)
    summary = df.mean().to_frame().T
    summary.index = ["Moyenne globale"]
    return df, summary