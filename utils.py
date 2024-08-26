import numpy as np
import cv2
import torch
import time

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2, timecost):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]
    
    cv2.putText(img1, f'FPS: {1/timecost:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }
def sp_lg(sp, lg, ref, tgt):
    """
    Function to perform SuperPoint + LightGlue pipeline.
    
    Args:
        sp: SuperPoint model
        lg: LightGlue model
        ref: reference image
        tgt: target image
    
    Returns:
        kpts0: keypoints of reference image
        kpts1: keypoints of target image
        time_det: time taken for detection
        time_mat: time taken for matching
    """
    
    ref = numpy_image_to_torch(ref)
    tgt = numpy_image_to_torch(tgt)
    
    tik = time.time()
    feats0 = sp.extract(ref.to(torch.device('cuda')))
    feats1 = sp.extract(tgt.to(torch.device('cuda')))
    tok = time.time()
    time_det = tok - tik
    
    tik = time.time()
    matches01 = lg({"image0": feats0, "image1": feats1})
    tok = time.time()
    time_mat = tok - tik
    
    time_total = time_det + time_mat
    
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]
    
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    mkpts_0, mkpts_1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
    if isinstance(mkpts_0, torch.Tensor):
        mkpts_0 = mkpts_0.cpu().numpy()
    if isinstance(mkpts_1, torch.Tensor):
        mkpts_1 = mkpts_1.cpu().numpy()
    
    if isinstance(tgt, torch.Tensor):
        tgt = tgt.cpu().numpy()
        tgt = np.transpose(tgt,(1,2,0))
        tgt = cv2.cvtColor(np.uint8(tgt*255), cv2.COLOR_RGB2BGR)

    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().numpy()
        ref = np.transpose(ref,(1,2,0))
        ref = cv2.cvtColor(np.uint8(ref*255), cv2.COLOR_RGB2BGR)
    
    return mkpts_0, mkpts_1, time_det, time_mat

def eloftr(model, ref, tgt):
    """
    Function to perform LoFTR pipeline.
    
    Args:
        model: LoFTR model
        ref: reference image
        tgt: target image
    
    Returns:
        kpts0: keypoints of reference image
        kpts1: keypoints of target image
        time_det: time taken for detection
        time_mat: time taken for matching
    """
    
    ref = cv2.resize(ref, (ref.shape[1]//32*32, ref.shape[0]//32*32))
    tgt = cv2.resize(tgt, (tgt.shape[1]//32*32, tgt.shape[0]//32*32))
    
    ref = torch.from_numpy(ref)[None][None].cuda()/255.
    tgt = torch.from_numpy(tgt)[None][None].cuda()/255.
    
    batch = {'image0': ref, 'image1': tgt}
    
    tik = time.time()
    with torch.no_grad():
        model(batch)
    tok = time.time()
    time_total = tok - tik
    
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    
    return mkpts0, mkpts1, 0, time_total