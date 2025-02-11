import ppsci
from ppsci.utils import logger
from omegaconf import DictConfig
import hydra
import paddle
from ppsci.data.dataset.stafnet_dataset import gat_lstmcollate_fn
import multiprocessing

def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.STAFNet(**cfg.MODEL) 
    train_dataloader_cfg = {
        "dataset": {
            "name": "STAFNetDataset",
            "file_path": cfg.DATASET.data_dir,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "seq_len": cfg.MODEL.seq_len,
            "pred_len": cfg.MODEL.pred_len,

        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "collate_fn": gat_lstmcollate_fn,
    }
    eval_dataloader_cfg= {
        "dataset": {
            "name": "STAFNetDataset",
            "file_path": cfg.EVAL.eval_data_path,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "seq_len": cfg.MODEL.seq_len,
            "pred_len": cfg.MODEL.pred_len,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "collate_fn": gat_lstmcollate_fn,
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        name="STAFNet_Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}
    
     # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Step(**cfg.TRAIN.lr_scheduler)()
    LEARNING_RATE = cfg.TRAIN.lr_scheduler.learning_rate
    optimizer = ppsci.optimizer.Adam(LEARNING_RATE)(model)
    output_dir = cfg.output_dir
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        ITERS_PER_EPOCH,
        eval_during_train=cfg.TRAIN.eval_during_train,
        seed=cfg.seed,
        validator=validator,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # train model
    solver.train()

def evaluate(cfg: DictConfig):
    """
    Validate after training an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains information about validation
    """
    model = ppsci.arch.STAFNet(**cfg.MODEL) 
    eval_dataloader_cfg= {
        "dataset": {
            "name": "STAFNetDataset",
            "file_path": cfg.EVAL.eval_data_path,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "seq_len": cfg.MODEL.seq_len,
            "pred_len": cfg.MODEL.pred_len,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "collate_fn": gat_lstmcollate_fn,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluate model
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="stafnet.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")

if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_example"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    multiprocessing.set_start_method("spawn")

    main()
