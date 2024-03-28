import os
import sys
sys.path.append(".")
import torch
from networks import *
from omegaconf import OmegaConf
import os
from torch.utils.data import DataLoader
from utils.data_utils import SE3Demo
from utils.loss_utils import double_geodesic_distance_between_poses
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import StepLR

def main(cfg):
    all_cfg = OmegaConf.load(f"config/{args.exp_name}/{args.pick_or_place}/config.json")
    cfg = all_cfg.mani
    cfg_seg = all_cfg.seg

    wd = os.path.join("experiments", args.exp_name, args.pick_or_place)
    os.makedirs(wd, exist_ok=True)
    demo_path = os.path.join("data", args.exp_name, args.pick_or_place, "demo.npz")

    demo = SE3Demo(demo_path, data_aug=cfg.data_aug, aug_methods=cfg.aug_methods, device=cfg.device)
    train_size = int(len(demo) * cfg.train_demo_ratio)
    test_size = len(demo) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(demo, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=True)

    policy = globals()[cfg.model](voxel_size=cfg.voxel_size, radius_threshold=cfg.radius_threshold, feature_point_radius=cfg.feature_point_radius).float().to(cfg.device)
    optm = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    scheduler = StepLR(optm, step_size=int(cfg.epoch/3), gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    policy_seg = globals()[cfg_seg.model](voxel_size=cfg_seg.voxel_size, radius_threshold=cfg_seg.radius_threshold).float().to(cfg_seg.device)
    policy_seg.load_state_dict(torch.load(os.path.join(wd, "segnet.pth")))
    policy_seg.eval()

    best_test_loss = 1e5

    for epoch in range(cfg.epoch):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epoch}")
        policy.train()

        for i, data in enumerate(progress_bar):
            optm.zero_grad()

            with torch.no_grad():
                ref_point = policy_seg(
                    {"xyz": data["xyz"], "rgb": data["rgb"]}, 
                    random_drop=False,
                )

            if cfg.ref_point == "seg_net":
                training_ref_point = ref_point
            elif cfg.ref_point == "gt":
                training_ref_point = data["seg_center"]

            output_pos, output_direction = policy(
                {"xyz": data["xyz"], "rgb": data["rgb"]}, 
                reference_point=training_ref_point, 
                distance_threshold=cfg.distance_threshold,
                random_drop=cfg.random_drop,
                train_pos=True,
                draw_pcd=cfg.draw_pcd,
                pcd_name=f"{i}",
                mask_part=cfg.mask_part,
            )

            pos_loss = loss_fn(output_pos, data["seg_center"])
            ori_loss = loss_fn(output_direction, data["axes"])

            if epoch < cfg.pos_warmup_epoch:
                loss = pos_loss
            else:
                loss = pos_loss + 0.1* ori_loss
            loss.backward()
            optm.step()

            with torch.no_grad():
                T1 = torch.zeros([data["axes"].shape[0], 4, 4]).to(cfg.device)
                T2 = torch.zeros_like(T1).to(cfg.device)
                T1[:, :3, :3] = data["axes"].reshape(data["axes"].shape[0], 3, 3).transpose(1,2)
                T1[:, :3, 3] = data["seg_center"]
                T1[:, 3, 3] = 1.
                T2[:, :3, :3] = output_direction.reshape(data["axes"].shape[0], 3, 3).transpose(1, 2)
                T2[:, :3, 3] = output_pos
                T2[:, 3, 3] = 1.
                t_loss, r_loss = double_geodesic_distance_between_poses(T1, T2, return_both=True)

            progress_bar.set_postfix(pos_loss=t_loss.item(), ori_loss=r_loss.item())

        policy.eval()
        with torch.no_grad():
            test_pos_loss = 0
            test_ori_loss = 0
            for batch_idx, data in enumerate(test_loader):
                output_pos, output_direction = policy(
                    {"xyz": data["xyz"], "rgb": data["rgb"]}, 
                    reference_point=data["seg_center"], 
                    distance_threshold=cfg.distance_threshold,
                    random_drop=cfg.random_drop,
                    train_pos=False,
                    draw_pcd=cfg.draw_pcd,
                    pcd_name=f"test_{batch_idx}",
                )
                pos_loss = loss_fn(output_pos, data["seg_center"])
                ori_loss = loss_fn(output_direction, data["axes"])

                T1 = torch.zeros([data["axes"].shape[0], 4, 4]).to(cfg.device)
                T2 = torch.zeros_like(T1).to(cfg.device)
                T1[:, :3, :3] = data["axes"].reshape(data["axes"].shape[0], 3, 3).transpose(1,2)
                T1[:, :3, 3] = data["seg_center"]
                T1[:, 3, 3] = 1.
                T2[:, :3, :3] = output_direction.reshape(data["axes"].shape[0], 3, 3).transpose(1, 2)
                T2[:, :3, 3] = output_pos
                T2[:, 3, 3] = 1.
                t_loss, r_loss = double_geodesic_distance_between_poses(T1, T2, return_both=True)
                test_pos_loss += t_loss.item()
                test_ori_loss += r_loss.item()

            test_pos_loss /= len(test_loader)
            test_ori_loss /= len(test_loader)
            print("Epoch: ", epoch, " test pos loss: ", test_pos_loss, " test ori loss: ", test_ori_loss)
            if test_pos_loss + test_ori_loss < best_test_loss:
                best_test_loss = test_pos_loss + test_ori_loss
                torch.save(policy.state_dict(), os.path.join(wd, f"maninet.pth"))
                print("Model saved!")

        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="mug")
    parser.add_argument('--pick_or_place', type=str, choices=["pick", "place"], default="pick")
    args = parser.parse_args()

    main(args)