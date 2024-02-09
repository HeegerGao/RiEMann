import os
import sys
sys.path.append(".")
import torch
from networks import *
from omegaconf import OmegaConf
import os
from torch.utils.data import DataLoader
from utils.data_utils import SE3Demo
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import StepLR


def main(args):
    all_cfg = OmegaConf.load(f"config/{args.exp_name}/{args.pick_or_place}.json")
    cfg = all_cfg.seg

    wd = os.path.join("experiments", args.exp_name, args.pick_or_place)
    os.makedirs(wd, exist_ok=True)
    demo_path = os.path.join("data", args.exp_name, args.pick_or_place, "demo.npz")

    demo = SE3Demo(demo_path, data_aug=cfg.data_aug, aug_methods=cfg.aug_methods, device=cfg.device)
    train_size = int(len(demo) * cfg.train_demo_ratio)
    test_size = len(demo) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(demo, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=True)

    policy = globals()[cfg.model](voxel_size=cfg.voxel_size, radius_threshold=cfg.radius_threshold).float().to(cfg.device)
    optm = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    scheduler = StepLR(optm, step_size=int(cfg.epoch/5), gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    best_test_loss = 1e5

    for epoch in range(cfg.epoch):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epoch}")
        policy.train()

        for i, data in enumerate(progress_bar):
            optm.zero_grad()

            output_pos = policy(
                {"xyz": data["xyz"], "rgb": data["rgb"]}, 
                random_drop=cfg.random_drop,
                draw_pcd=cfg.draw_pcd,
                pcd_name=f"{i}",
                mask_part=cfg.mask_part,
            )
            loss = loss_fn(output_pos, data["seg_center"])
            loss.backward()
            optm.step()

            t_loss = torch.sqrt(torch.sum(torch.sqrt((output_pos-data["seg_center"]) ** 2), dim=1)).mean()
            progress_bar.set_postfix(loss=t_loss.item())

        policy.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_idx, data in enumerate(test_loader):
                output_pos = policy(
                    {"xyz": data["xyz"], "rgb": data["rgb"]}, 
                    random_drop=False,
                    draw_pcd=cfg.draw_pcd,
                    pcd_name=f"test_{batch_idx}",
                )
                t_loss = torch.sqrt(torch.sum(torch.sqrt((output_pos-data["seg_center"]) ** 2), dim=1)).mean()

                test_loss += t_loss.item()
            test_loss /= len(test_loader)
            print("Epoch: ", epoch, " seg test loss: ", test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(policy.state_dict(), os.path.join(wd, f"segnet.pth"))
                print("Model saved!")

        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="mug")
    parser.add_argument('--pick_or_place', type=str, choices=["pick", "place"], default="pick")
    args = parser.parse_args()

    main(args)