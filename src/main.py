import os
import argparse
from tqdm import tqdm
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple, Callable

import torch
from torch.utils.data import DataLoader

from Configs.builder import get_configs
from Losses.loss import loss
from Models.builder import get_models
from Datasets.builder import get_datasets
from Models.model_mamba import GMMFormer_Mamba_Net
from Opts.builder import get_opts
from Losses.builder import get_losses
from Opts.optimization import BertAdam
from Validations.builder import get_validations

from Utils.basic_utils import AverageMeter
from Utils.utils import set_seed, set_log, gpu, save_ckpt, load_ckpt
from Validations.validations import validations

parser = argparse.ArgumentParser(
    description="Partially Relevant Video Retrieval"
)
parser.add_argument(
    "-d",
    "--dataset_name",
    default="cha",
    type=str,
    metavar="DATASET",
    help="dataset name",
    choices=["tvr", "act", "cha"],
)
parser.add_argument("--gpu", default="0", type=str, help="specify gpu device")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--resume", default="", type=str)
parser.add_argument("--loss_factor", default=None, type=str)
parser.add_argument("--lr", default=None, type=float)
parser.add_argument(
    "--save_best",
    action="store_true",
    help="Save the best models from previous iterations",
)
args = parser.parse_args()


def load_param(cfg: Dict[str, Any]) -> None:
    param_mapping: Dict[str, Callable[[str], Any]] = {
        "loss_factor": lambda x: [float(i) for i in x.split(",")],
        "lr": float,
    }
    for param_name, converter in param_mapping.items():
        value = getattr(args, param_name)
        if value is not None:
            try:
                cfg[param_name] = converter(value)
            except (ValueError, TypeError):
                logging.error(
                    f"Invalid value for {param_name}: {value}. "
                    + "Using default instead."
                )


def train_one_epoch(
    epoch: int,
    train_loader: DataLoader,
    model: GMMFormer_Mamba_Net,
    criterion: loss,
    cfg: Dict[str, Any],
    optimizer: BertAdam,
    logger: logging,
) -> float:
    if epoch >= cfg["hard_negative_start_epoch"]:
        criterion.cfg["use_hard_negative"] = True
    else:
        criterion.cfg["use_hard_negative"] = False

    loss_meter = AverageMeter()
    model.train()

    train_bar = tqdm(
        train_loader,
        desc=f"epoch {epoch}",
        total=len(train_loader),
        unit="batch",
        dynamic_ncols=True,
    )
    for idx, batch in enumerate(train_bar):
        try:
            batch = gpu(batch)
            optimizer.zero_grad()
            input_list = model(batch)
            loss = criterion(input_list, batch)

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
        except Exception as e:
            logger.error(
                "Errors occurred during training, "
                + f"at epoch: {epoch}, batch: {idx}"
            )
            logger.error(str(e))
            exit(-1)

        train_bar.set_description(
            f"exp: {cfg['model_name']} epoch:{epoch:2d} "
            + f"iter:{idx:3d} loss:{loss:.4f}"
        )
    return loss_meter.avg


def val_one_epoch(
    epoch: int,
    context_dataloader: DataLoader,
    query_eval_loader: DataLoader,
    model: GMMFormer_Mamba_Net,
    val_criterion: validations,
    cfg: Dict[str, Any],
    optimizer: BertAdam,
    best_val: List[float],
    loss_meter: float,
    logger: logging,
) -> Tuple[List[float], List[float], bool]:
    val_meter = val_criterion(model, context_dataloader, query_eval_loader)
    if val_meter[4] > best_val[4]:
        es = False
        sc = "New Best Model !!!"
        best_val = val_meter
        if args.save_best:
            formatted_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(os.path.dirname(cfg["ckpt_path"]), exist_ok=True)
            save_ckpt(
                model,
                optimizer,
                cfg,
                os.path.join(
                    cfg["ckpt_path"], f"{best_val[4]:.2f}_{formatted_now}.ckpt"
                ),
                epoch,
                best_val,
            )
    else:
        es = True
        sc = "A Relative Failure Epoch"

    logger.partition()
    logger.info(f"Epoch: {epoch:2d}    {sc}")
    logger.info(f"Average Loss: {loss_meter:.4f}")
    logger.info(f"R@1: {val_meter[0]:.1f}")
    logger.info(f"R@5: {val_meter[1]:.1f}")
    logger.info(f"R@10: {val_meter[2]:.1f}")
    logger.info(f"R@100: {val_meter[3]:.1f}")
    logger.info(f"Rsum: {val_meter[4]:.1f}")
    logger.info(
        f"Best: R@1: {best_val[0]:.1f} R@5: {best_val[1]:.1f} "
        + f"R@10: {best_val[2]:.1f} R@100: {best_val[3]:.1f} "
        + f"Rsum: {best_val[4]:.1f}"
    )
    logger.partition()
    return val_meter, best_val, es


def validation(
    context_dataloader: DataLoader,
    query_eval_loader: DataLoader,
    model: GMMFormer_Mamba_Net,
    val_criterion: validations,
    logger: logging,
    resume: str,
) -> None:
    val_meter = val_criterion(model, context_dataloader, query_eval_loader)
    logger.partition()
    logger.info(f"Testing from: {resume}")
    logger.info(f"R@1: {val_meter[0]:.1f}")
    logger.info(f"R@5: {val_meter[1]:.1f}")
    logger.info(f"R@10: {val_meter[2]:.1f}")
    logger.info(f"R@100: {val_meter[3]:.1f}")
    logger.info(f"Rsum: {val_meter[4]:.1f}")
    logger.partition()


def main() -> None:
    cfg: Dict[str, Any] = get_configs(args.dataset_name)
    load_param(cfg)

    logger: logging = set_log(cfg["model_root"], "log.txt")
    logger.info(
        f"Partially Relevant Video Retrieval Training: {cfg['dataset_name']}"
    )

    set_seed(cfg["seed"])
    logger.info(f"set seed: {cfg['seed']}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for i in range(torch.cuda.device_count()):
        logger.info(f"used gpu{i}: {torch.cuda.get_device_name(i)}")

    logger.info("Hyper Parameter ......")
    logger.info(cfg)

    logger.info("Loading Data ......")
    (
        cfg,
        train_loader,
        context_dataloader,
        query_eval_loader,
        test_context_dataloader,
        test_query_eval_loader,
    ) = get_datasets(cfg)

    logger.info("Loading Model ......")
    model: GMMFormer_Mamba_Net = get_models(cfg)

    current_epoch: int = -1
    es_cnt: int = 0
    best_val: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    if args.resume != "":
        logger.info(f"Resume from {args.resume}")
        _, model_state_dict, __, current_epoch, best_val = load_ckpt(
            args.resume
        )
        model.load_state_dict(model_state_dict, strict=False)
    model = model.cuda()

    criterion = get_losses(cfg)
    val_criterion: validations = get_validations(cfg)

    if args.eval:
        if args.resume == "":
            logger.error("No trained ckpt load !!!")
        else:
            with torch.no_grad():
                validation(
                    test_context_dataloader,
                    test_query_eval_loader,
                    model,
                    val_criterion,
                    logger,
                    args.resume,
                )
        exit(0)

    optimizer: BertAdam = get_opts(cfg, model, train_loader)

    for epoch in range(current_epoch + 1, cfg["n_epoch"]):
        loss_meter = train_one_epoch(
            epoch, train_loader, model, criterion, cfg, optimizer, logger
        )
        with torch.no_grad():
            _, best_val, es = val_one_epoch(
                epoch,
                context_dataloader,
                query_eval_loader,
                model,
                val_criterion,
                cfg,
                optimizer,
                best_val,
                loss_meter,
                logger,
            )

        if not es:
            es_cnt = 0
        else:
            es_cnt += 1
            if cfg["max_es_cnt"] != -1 and es_cnt > cfg["max_es_cnt"]:
                logger.info("Early Stop !!!")
                exit(0)


if __name__ == "__main__":
    main()
