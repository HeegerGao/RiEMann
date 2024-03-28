import sys
sys.path.append(".")
import os
import torch
from networks import *
from omegaconf import OmegaConf
from utils.data_utils import downsample_table
import argparse
import numpy as np
from utils.utils import modified_gram_schmidt

def main(args):
    all_cfg = OmegaConf.load(f"config/{args.exp_name}/{args.pick_or_place}/config.json")
    cfg_seg = all_cfg.seg
    cfg_mani = all_cfg.mani

    wd = os.path.join("experiments", args.exp_name, args.pick_or_place, args.setting)
    pcd_path = os.path.join("data", args.exp_name, args.pick_or_place, f"{args.setting}.npz")
    pcd = np.load(pcd_path)

    input_xyz = torch.tensor(pcd["xyz"]).float().unsqueeze(0).to(cfg_seg.device)
    input_rgb = torch.tensor(pcd["rgb"]).float().unsqueeze(0).to(cfg_seg.device)
    
    model_dir = os.path.join("experiments", args.exp_name, args.pick_or_place)
    policy_seg = globals()[cfg_seg.model](voxel_size=cfg_seg.voxel_size, radius_threshold=cfg_seg.radius_threshold).float().to(cfg_seg.device)
    policy_seg.load_state_dict(torch.load(os.path.join(model_dir, "segnet.pth")))
    policy_seg.eval()

    policy_mani = globals()[cfg_mani.model](voxel_size=cfg_mani.voxel_size, radius_threshold=cfg_mani.radius_threshold).float().to(cfg_mani.device)
    policy_mani.load_state_dict(torch.load(os.path.join(model_dir, "maninet.pth")))
    policy_mani.eval()

    assert cfg_seg.device == cfg_mani.device, "Device mismatch between segmentation and manipulation networks!"

    # preprossing input to keep the same as training
    data = {
        "xyz": input_xyz[0],
        "rgb": input_rgb[0],
    }
    data = downsample_table(data)

    xyz = data["xyz"].unsqueeze(0).to(cfg_seg.device)
    rgb = data["rgb"].unsqueeze(0).to(cfg_seg.device)

    with torch.no_grad():
        ref_point = policy_seg(
            {"xyz": xyz, "rgb": rgb}, 
            draw_pcd=True,
            pcd_name=args.setting,
        )
        output_pos, output_direction, _, _, _ = policy_mani(
            {"xyz": xyz, "rgb": rgb}, 
            reference_point=ref_point, 
            distance_threshold=cfg_mani.distance_threshold,
            save_ori_feature = True,
            draw_pcd=True,
            pcd_name=args.setting,
        )
        out_dir_schmidt = modified_gram_schmidt(output_direction.reshape(-1, 3).T, to_cuda=True)

    pred_pos = output_pos.detach().cpu().numpy().reshape(3)
    pred_rot = out_dir_schmidt.detach().cpu().numpy()

    result_path = os.path.join(wd, f"{args.setting}_pred_pose.npz")
    np.savez(result_path,
             pred_pos=pred_pos,
             pred_rot=pred_rot,
            )
    print(f"Result saved to: {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_name', type=str, default="mug")
    parser.add_argument('-pick_or_place', type=str, choices=["pick", "place"], default="pick")
    parser.add_argument('-setting', type=str, default='newpose')
    args = parser.parse_args()

    main(args)