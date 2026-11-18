from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.logging import MMLogger, print_log
import torch
import numpy as np
import cv2
import math


# ============ 辅助函数：计算额外指标 ============

def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide boundaries.
    """
    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1 if seg.ndim == 3 else True

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w or height > h or abs(ar1 - ar2) > 0.01
    ), f"Can't convert {w}x{h} seg to {width}x{height} bmap."

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = int(np.ceil(y * height / h))
                    i = int(np.ceil(x * width / w))
                    bmap[j, i] = 1

    return bmap


def compute_boundary_f_measure(pred_mask, gt_mask, bound_th=0.008):
    """
    计算边界F-measure (shape accuracy)
    
    Args:
        pred_mask: [H, W] binary mask (0/1)
        gt_mask: [H, W] binary mask (0/1)
        bound_th: boundary thickness threshold
        
    Returns:
        float: F-measure score
    """
    try:
        if not isinstance(pred_mask, np.ndarray):
            pred_mask = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else np.array(pred_mask)
        if not isinstance(gt_mask, np.ndarray):
            gt_mask = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else np.array(gt_mask)
        
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        bound_pix = (
            bound_th
            if bound_th >= 1
            else np.ceil(bound_th * np.linalg.norm(pred_mask.shape))
        )

        # Get the pixel boundaries of both masks
        pred_boundary = _seg2bmap(pred_mask)
        gt_boundary = _seg2bmap(gt_mask)

        from skimage.morphology import disk

        # Dilate boundaries
        pred_dil = cv2.dilate(pred_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
        gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

        # Get the intersection
        gt_match = gt_boundary * pred_dil
        pred_match = pred_boundary * gt_dil

        # Area of the intersection
        n_pred = np.sum(pred_boundary)
        n_gt = np.sum(gt_boundary)

        # Compute precision and recall
        if n_pred == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_pred > 0 and n_gt == 0:
            precision = 0
            recall = 1
        elif n_pred == 0 and n_gt == 0:
            precision = 1
            recall = 1
        else:
            precision = np.sum(pred_match) / float(n_pred)
            recall = np.sum(gt_match) / float(n_gt)

        # Compute F measure
        if precision + recall == 0:
            F = 0
        else:
            F = 2 * precision * recall / (precision + recall)

        return float(F)
    
    except Exception as e:
        print(f"[Warning] Boundary F-measure computation failed: {e}")
        return 0.0


def compute_location_score(pred_mask, gt_mask):
    """
    计算位置得分：基于质心距离的归一化得分
    
    Args:
        pred_mask: [H, W] binary mask
        gt_mask: [H, W] binary mask
        
    Returns:
        float: location score (normalized centroid distance)
    """
    try:
        if not isinstance(pred_mask, np.ndarray):
            pred_mask = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else np.array(pred_mask)
        if not isinstance(gt_mask, np.ndarray):
            gt_mask = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else np.array(gt_mask)
        
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        H, W = pred_mask.shape[:2]
        
        # 获取预测mask的质心
        pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(pred_contours) == 0 or len(gt_contours) == 0:
            # 如果其中一个mask为空，返回最大距离
            return 1.0
        
        pred_bigc = max(pred_contours, key=cv2.contourArea)
        gt_bigc = max(gt_contours, key=cv2.contourArea)
        
        pred_centroid = pred_bigc.mean(axis=0)[0]
        gt_centroid = gt_bigc.mean(axis=0)[0]
        
        # 计算欧氏距离
        centroid_distance = np.linalg.norm(pred_centroid - gt_centroid)
        
        # 归一化到图像对角线长度
        diagonal = np.sqrt(H**2 + W**2)
        location_score = centroid_distance / diagonal
        
        return float(location_score)
    
    except Exception as e:
        print(f"[Warning] Location score computation failed: {e}")
        return 1.0  # 返回最差得分


@METRICS.register_module()
class SegMetric(BaseMetric):
    """计算 IoU 和 Dice 的简单 metric，兼容 MMEngine"""

    def __init__(self, iou_metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.iou_metrics = iou_metrics or ["IoU", "Dice"]
        self.results = []

    def process(self, data_batch, data_samples):

        gt_masks_all = data_batch["data"].get("masks", None)
        bs = len(gt_masks_all) # 1
        # print(f"bs{bs}")
        num_obj = gt_masks_all[0].shape[0]

        for i in range(bs):
            gt_masks = gt_masks_all[i]
            pred_masks = data_samples[i]["pred_masks"]
            for j in range(num_obj):
                gt_mask_i = gt_masks[j]
                pred_mask_i = pred_masks[j]
                pred = pred_mask_i.float().to(pred_mask_i.device)
                gt = gt_mask_i.float().to(pred_mask_i.device)
                inter = (pred * gt).sum()
                union = (pred + gt - pred * gt).sum()
                iou = inter / (union + 1e-6)
                dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
                
                # wrz: 计算新增指标 - Location Score 和 Shape Accuracy
                location_score = compute_location_score(pred_mask_i, gt_mask_i)
                shape_acc = compute_boundary_f_measure(pred_mask_i, gt_mask_i)
                
                # print(iou)
                self.results.append(dict(
                    IoU=iou.item(), 
                    Dice=dice.item(),
                    location_score=location_score,
                    shape_acc=shape_acc
                ))
            
    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()

        if not results:
            print_log("⚠️ No results collected in SegMetric!", logger=logger)
            return {}

        ious = [r["IoU"] for r in results]
        dices = [r["Dice"] for r in results]
        mean_iou = sum(ious) / len(ious)
        mean_dice = sum(dices) / len(dices)
        
        # wrz: 计算新增指标的平均值
        location_scores = [r.get("location_score", 1.0) for r in results]
        shape_accs = [r.get("shape_acc", 0.0) for r in results]
        
        mean_location_score = sum(location_scores) / len(location_scores) if location_scores else 1.0
        mean_shape_acc = sum(shape_accs) / len(shape_accs) if shape_accs else 0.0

        # ✅ 使用 MMEngine 的日志系统
        print_log("=" * 40, logger=logger)
        print_log(f"✅ Mean IoU:  {mean_iou:.4f}", logger=logger)
        print_log(f"✅ Mean Dice: {mean_dice:.4f}", logger=logger)
        print_log(f"📍 Mean Location Score: {mean_location_score:.4f} (lower is better)", logger=logger)
        print_log(f"🎯 Mean Shape Accuracy: {mean_shape_acc:.4f}", logger=logger)
        print_log("=" * 40, logger=logger)

        return dict(
            mean_IoU=mean_iou, 
            mean_Dice=mean_dice,
            mean_location_score=mean_location_score,
            mean_shape_acc=mean_shape_acc
        )
