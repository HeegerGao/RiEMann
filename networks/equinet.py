import torch
import os
from networks.se3_backbone import SE3Backbone, ExtendedModule
from networks.se3_transformer.model.fiber import Fiber
from utils.data_utils import seg_pointcloud, random_dropout, mask_part_point_cloud

from utils.vis import save_pcd_as_pcd
from utils.data_utils import get_heatmap


class SE3SegNet(ExtendedModule):
    def __init__(self, voxelize=True, voxel_size=0.03, radius_threshold=0.1):
        super().__init__()
        self.backbone = SE3Backbone(
            fiber_out=Fiber({
                "0": 1, # one heatmap
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )

    def forward(self, inputs, draw_pcd=False, random_drop=False, pcd_name=None, mask_part=False):
        bs = inputs["xyz"].shape[0]
        new_inputs = {
            "xyz": [],
            "rgb": [],
            "feature": []
        }
        for i in range(bs):
            data = {
                    "xyz": inputs["xyz"][i],
                    "rgb": inputs["rgb"][i],
                }
            if random_drop:
                data = random_dropout(data["xyz"], data["rgb"])
            if mask_part:
                data = mask_part_point_cloud(data["xyz"], data["rgb"])
            
            new_inputs["xyz"].append(data["xyz"])
            new_inputs["rgb"].append(data["rgb"])
            new_inputs["feature"].append(new_inputs["rgb"][i])
        inputs = new_inputs
        backbone_output = self.backbone(inputs)

        xyz = backbone_output["xyz"]
        feature = backbone_output["feature"]

        output_pos = torch.zeros([len(xyz), 3]).to(self.backbone.device)
        for i in range(len(xyz)):
            pos_weight = torch.nn.functional.softmax(feature[i].reshape(-1, 1), dim=0).squeeze()
            output_pos[i] = (xyz[i].T * pos_weight).T.sum(dim=0)

        if draw_pcd:
            os.makedirs("pcd/seg", exist_ok=True)
            for i in range(len(xyz)):
                save_pcd_as_pcd(xyz[i], backbone_output["given_graph"]["raw_node_feats"][i][:, :3], save_file=f"pcd/seg/original_{pcd_name}_{i}.pcd")
                save_pcd_as_pcd(xyz[i], pos_weight.unsqueeze(-1).repeat(1, 3)/torch.max(pos_weight), save_file=f"pcd/seg/pos_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

        return output_pos


class SE3ManiNet(ExtendedModule):
    def __init__(self, voxelize=True, voxel_size=0.01, radius_threshold=0.12, feature_point_radius=0.02):
        super().__init__()
        self.pos_net = SE3Backbone(
            fiber_out=Fiber({
                "0": 1, # one heatmap
            }),
            num_layers= 4,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )

        self.ori_net = SE3Backbone(
            fiber_out=Fiber({
                "1": 3,
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
            voxel_size= voxel_size,
            radius_threshold=radius_threshold,
        )
        self.feature_point_radius = feature_point_radius

    def forward(self, inputs, train_pos=False, reference_point=None, distance_threshold=0.3, random_drop=False, draw_pcd=False, pcd_name=None, mask_part=False, save_ori_feature=False):
        bs = inputs["xyz"].shape[0]
        new_inputs = {
            "xyz": [],
            "rgb": [],
            "feature": []
        }
        gt_heatmaps = []
        for i in range(bs):
            if draw_pcd:
                os.makedirs("pcd/mani", exist_ok=True)
                distances = torch.norm(inputs["xyz"][i] - reference_point[i], dim=1)
                closest_point_idx = torch.argmin(distances)
                save_pcd_as_pcd(inputs["xyz"][i], inputs["rgb"][i], save_file=f"pcd/mani/original_{pcd_name}_{i}.pcd")

                gt_heatmaps.append(get_heatmap(inputs["xyz"][i], closest_point_idx, std_dev=0.015, max_value=1).to(self.pos_net.device))
                save_pcd_as_pcd(inputs["xyz"][i], gt_heatmaps[-1].unsqueeze(-1).repeat(1, 3)/torch.max(gt_heatmaps[-1]), save_file=f"pcd/mani/gt_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

            if reference_point != None:
                data = seg_pointcloud(inputs["xyz"][i], inputs["rgb"][i], reference_point[i], distance=distance_threshold)
            else:
                data = {
                    "xyz": inputs["xyz"][i],
                    "rgb": inputs["rgb"][i],
                }
            if random_drop:
                data = random_dropout(data["xyz"], data["rgb"])
            if mask_part:
                data = mask_part_point_cloud(data["xyz"], data["rgb"])
            new_inputs["xyz"].append(data["xyz"])
            new_inputs["rgb"].append(data["rgb"])
            new_inputs["feature"].append(new_inputs["rgb"][i])
        inputs = new_inputs

        # pos
        if train_pos:
            seg_output = self.pos_net(inputs)
            xyz = seg_output["xyz"]
            feature = seg_output["feature"]
            pos_weights = []

            output_pos = torch.zeros([len(xyz), 3]).to(self.device)
            for i in range(len(xyz)):
                if draw_pcd:
                    save_pcd_as_pcd(xyz[i], feature[i][:, 0].clone().unsqueeze(-1).repeat(1, 3)/torch.max(feature[i][:, 0].clone()), save_file=f"pcd/mani/pos_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

                pos_weight = torch.nn.functional.softmax(feature[i].reshape(-1, 1), dim=0).squeeze()
                output_pos[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
                pos_weights.append(pos_weight)
        else:
            with torch.no_grad():
                seg_output = self.pos_net(inputs)
                xyz = seg_output["xyz"]
                feature = seg_output["feature"]
                pos_weights = []

                output_pos = torch.zeros([len(xyz), 3]).to(self.device)
                for i in range(len(xyz)):
                    if draw_pcd:
                        save_pcd_as_pcd(xyz[i], feature[i][:, 0].clone().unsqueeze(-1).repeat(1, 3)/torch.max(feature[i][:, 0].clone()), save_file=f"pcd/mani/pos_heatmap_{pcd_name}_{i}.pcd", draw_heatmap=True)

                    pos_weight = torch.nn.functional.softmax(feature[i].reshape(-1, 1), dim=0).squeeze()
                    output_pos[i] = (xyz[i].T * pos_weight).T.sum(dim=0)
                    pos_weights.append(pos_weight)

        if draw_pcd:
            for i in range(len(xyz)):
                distances = torch.norm(xyz[i] - reference_point[i], dim=1)
                closest_point_idx = torch.argmin(distances)
                save_pcd_as_pcd(xyz[i], seg_output["given_graph"]["raw_node_feats"][i][:, :3], save_file=f"pcd/mani/ball_{pcd_name}_{i}.pcd")

        ori_output = self.ori_net(inputs)
        xyz = ori_output["xyz"]
        feature = ori_output["feature"]    # 3*3 = 9
        output_ori = torch.zeros([len(xyz), 9]).to(self.device)

        if save_ori_feature:
            for i in range(len(xyz)):
                torch.save(feature[i].cpu(), f"pcd/mani/ori_feature_{pcd_name}_{i}.pt")

        for i in range(bs):
            newdata = seg_pointcloud(xyz[i], xyz[i], reference_point=output_pos[i], distance=self.feature_point_radius, extra_data={"feature": feature[i]})
            if newdata["xyz"].shape[0] == 0:
                # use the pos point
                output_ori[i] = (feature[i].T * pos_weights[i].detach()).T.sum(dim=0)
            else:
                output_ori[i] = newdata["feature"].mean(dim=0)

        for i in range(3):
            output_ori[:, 3*i:3*(i+1)] /= (torch.norm(output_ori[:, 3*i:3*(i+1)].clone(), dim=1).unsqueeze(1) + 1e-8)
        return output_pos, output_ori
    
