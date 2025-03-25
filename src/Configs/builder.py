import os
import yaml


def get_configs(dataset_name):
    cfg = {}
    if dataset_name == "tvr":
        cfg["dataset_name"] = "tvr"
        cfg["visual_feature"] = "i3d_resnet"
    elif dataset_name == "cha":
        cfg["dataset_name"] = "charades"
        cfg["visual_feature"] = "i3d_rgb_lgi"
    elif dataset_name == "act":
        cfg["dataset_name"] = "activitynet"
        cfg["visual_feature"] = "i3d"

    cfg["model_name"] = "MamFusion"
    cfg["seed"] = 9527
    cfg["root"] = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # cfg["data_root"] = os.path.join(cfg["root"], "data", "netdisk")
    cfg["data_root"] = os.path.join("/home/ac/data/dataset", "netdisk")

    cfg["collection"] = cfg["dataset_name"]
    cfg["map_size"] = 32
    cfg["clip_scale_w"] = 0.7
    cfg["frame_scale_w"] = 0.3

    cfg["model_root"] = os.path.join(
        cfg["root"], "results", cfg["dataset_name"], cfg["model_name"]
    )
    cfg["ckpt_path"] = os.path.join(cfg["model_root"], "ckpt")

    if not os.path.exists(cfg["model_root"]):
        os.makedirs(cfg["model_root"], exist_ok=True)
    if not os.path.exists(cfg["ckpt_path"]):
        os.makedirs(cfg["ckpt_path"], exist_ok=True)

    with open(
        os.path.join(cfg["root"], "src", "Configs", f"{dataset_name}.yaml"),
        "r",
    ) as yaml_file:
        config = yaml.safe_load(yaml_file)
        cfg.update(dict(config))

    cfg["num_workers"] = 14 if cfg["no_core_driver"] else cfg["num_workers"]
    cfg["pin_memory"] = not cfg["no_pin_memory"]

    return cfg
