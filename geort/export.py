# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os 
from pathlib import Path
import json
from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.path import to_package_root, get_checkpoint_root
from geort.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''
    def __init__(self, model_path, config_path):
        config = load_json(config_path)
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        # print(keypoint_info["joint"])  # Commented out to reduce verbose output
        self.human_ids = keypoint_info["human_id"]
        self.model = IKModel(keypoint_joints=keypoint_info["joint"]).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit) # GeoRT will do normalization.

    def forward(self, keypoints):
        # keypoints: [N, 3]
        keypoints = keypoints[self.human_ids] # extract.
        joint_normalized = self.model.forward(torch.from_numpy(keypoints).unsqueeze(0).reshape(1, -1, 3).float().cuda())
        joint_raw = self.qpos_normalizer.unnormalize(joint_normalized.detach().cpu().numpy())
        return joint_raw[0]


def load_model(tag='', epoch=0, use_best=True):
    '''
        Loading API with best model preference.
        
        Args:
            tag: checkpoint tag to search for
            epoch: specific epoch to load (0 means latest)
            use_best: if True, prefer best.pth over last.pth when epoch=0
    '''
    checkpoint_root = get_checkpoint_root()
    all_checkpoints = os.listdir(checkpoint_root)
    
    checkpoint_name = ''
    for checkpoint in all_checkpoints:
        if tag in checkpoint:
            checkpoint_name = checkpoint
            break 

    checkpoint_root = Path(checkpoint_root) / checkpoint_name
    
    if epoch > 0:
        model_path = checkpoint_root / f"epoch_{epoch}.pth"
    else:
        # Try to load best model first if use_best is True
        if use_best and (checkpoint_root / "best.pth").exists():
            model_path = checkpoint_root / "best.pth"
            # Load training history to show best model info
            history_path = checkpoint_root / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    print(f"Loading best model from epoch {history['best_epoch']} with validation loss: {history['best_val_loss']:.4f}")
        else:
            model_path = checkpoint_root / "last.pth"
    
    config_path = checkpoint_root / "config.json"
    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path)

if __name__ == '__main__':
    # load the model in one line.
    load_model(tag="allegro_last")