"""
@Description :   Matching keypoints between a reference image and a video using xfeat and mnn or superpoint and lightglue
@Author      :   Xubo Luo 
@Time        :   2024/08/22 09:29:30
"""

import numpy as np
import time
import cv2
import torch
import argparse

from copy import deepcopy
from xfeat.xfeat import XFeat
from lightglue import LightGlue, SuperPoint
from efficientloftr.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

from utils import warp_corners_and_draw_matches, sp_lg, eloftr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, help='Path to the reference image', default='assets/groot/groot.jpg')
    parser.add_argument('--tgt', type=str, help='Path to the target video', default='assets/groot/groot.mp4')
    parser.add_argument('--method', type=str, help='Method to use for image matching (xfeat+mnn, xfeat+lg, sp+lg, loftr)', default='xfeat+mnn')
    parser.add_argument('--save_path', type=str, help='Path to save the output video', default='output.mp4')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # --- Load the selected method --- #
    method = args.method
    if method == 'xfeat+mnn':
        xfeat = XFeat()
        print(f'Load xfeat to {xfeat.dev}')
    elif method == 'sp+lg':
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
        print(f'Load superpoint and lightglue to {device}')
    elif method == 'loftr':
        _default_cfg = deepcopy(full_default_cfg)
        loftr = LoFTR(config=_default_cfg)
        loftr.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
        loftr = reparameter(loftr)
        loftr = loftr.eval().to(device)
    elif method == 'xfeat+lg':
        xfeat = XFeat()
        print(f'Load xfeat and lighterglue to {xfeat.dev}')
    else:
        raise ValueError(f'Unknown method: {method}')
    
    
    # --- Load the reference image and target video --- #
    ref = cv2.imread(args.ref)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    tgt = cv2.VideoCapture(args.tgt)
    
    
    
    # --- Extract keypoints and match them --- #
    if tgt.isOpened():
        fps = tgt.get(cv2.CAP_PROP_FPS)
        width = int(tgt.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(tgt.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ref = cv2.resize(ref, (width, height))
        print(f'Width: {width}, Height: {height}')
        
        # --- Save the output video --- #
        if args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.save_path, fourcc, 30, (width*2, height))
        
        while True:
            ret, frame = tgt.read()
            
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if method == 'xfeat+mnn':
                mkpts_0, mkpts_1, time_det, time_mat = xfeat.match_xfeat(ref, frame, top_k = 4096)
            elif method == 'sp+lg':
                mkpts_0, mkpts_1, time_det, time_mat = sp_lg(extractor, matcher, ref, frame)
            elif method == 'loftr':
                mkpts_0, mkpts_1, time_det, time_mat = eloftr(loftr, ref, frame)
            elif method == 'xfeat+lg':
                mkpts_0, mkpts_1, time_det, time_mat = xfeat.detect_match_lighterglue(ref, frame)
                
            time_total = time_det + time_mat
            
            canvas = warp_corners_and_draw_matches(mkpts_1, mkpts_0, frame, ref, time_total)
            
            cv2.imshow('Video', canvas[..., ::-1])
            
            if args.save_path:
                save_frame = canvas[..., ::-1]
                save_frame = np.uint8(save_frame)
                out.write(save_frame)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        tgt.release()
        
        if args.save_path:
            out.release()
            print(f'Output video saved at {args.save_path}')
        
    cv2.destroyAllWindows()
    

        

        