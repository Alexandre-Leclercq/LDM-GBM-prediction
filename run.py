import argparse, os, sys, datetime, glob, importlib, csv
from functools import partial
import time
import pandas as pd
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning import Trainer

from omegaconf import OmegaConf

from utils.autoloading import instantiate_from_config
from utils.compute_latent_std import compute_latent_std


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of running test",
    )
    parser.add_argument(
        "--checkpoint-test",
        type=str,
        default="",
        help="use a specific checkpoint for testing",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


"""
Aimed to be used as a script
"""
if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:  # We can't start a new training and resume a training at the same time
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    if opt.resume:  # if we resume a training
        if not os.path.exists(opt.resume):  # we make sure the training exist
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt  # resume the model from the last checkpoint
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:  # if we train a new model
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)  # set dirname of logdir as date+name+postfix

    ckptdir = os.path.join(logdir, "checkpoints")  # checkpoints path
    cfgdir = os.path.join(logdir, "configs")  # configs path

    # import config model training
    configs = [OmegaConf.load(cfg) for cfg in opt.base]  # load all the config files
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)  # merge all the configs in one config
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config


    """
    -----   loading data    -----
    """
    seed_everything(config.data.params.seed, workers=True)
    data = instantiate_from_config(config.data)

    # if we want to compute the std_norm of latent space and that it hasn't been done before.
    # TODO should probably be removed
    if (hasattr(config.model, 'compute_std_norm') and
            config.model.compute_std_norm and
            not hasattr(config.model.params, 'std_source') and
            not hasattr(config.model.params, 'std_target')):
        print(f'{5*"="}\tpreparing computation for latent space std\t{5*"="}')
        print('loading model')
        # we load the model to do the computation with vae_source and vae_target
        model = instantiate_from_config(config.model)

        print('instantiate database')
        data.setup('latent')
        train_dataloader = data.train_dataloader()

        print("compute std latent space source")
        latent_source_std = compute_latent_std(model.vae_source, train_dataloader, idx_data='preMRI')
        print(latent_source_std)

        print("compute std latent space target")
        latent_target_std = compute_latent_std(model.vae_target, train_dataloader, idx_data='postMRI')
        print(latent_target_std)

        # we add the computed
        config.model.std_source = latent_source_std.item()
        config.model.std_target = latent_target_std.item()

    """    
    -----   loading model    -----
    """
    model = instantiate_from_config(config.model)

    """
    -----   setting logger    -----
    """
    trainer_kwargs = dict()

    if opt.train:
        # default logger configs
        default_logger_cfgs = {
            "tensorboard": {
                "target": "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir,
                }
            },
        }

        default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        default_modelckpt_cfg = {
            "target": "run.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "utils.callbacks.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "data_module": data
                }
            },
            "image_logger": {
                "target": "utils.callbacks.ImageLogger",
                "params": {
                    "batch_frequency_train": 500,
                    "batch_frequency_val": 50,
                    "max_images": 4,
                    "clamp": True,
                    "frequence_unit": config.image_logger.frequence_unit if hasattr(config, 'image_logger') else 'batch',
                    "rescale": 'tanh_range' in config.data.params and config.data.params.tanh_range
                }
            },
            "learning_rate_logger": {
                "target": "run.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "utils.callbacks.CUDACallback"
            },
        }
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        callbacks_cfg = default_callbacks_cfg

        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    """
    -----   setting trainer    -----
    """
    trainer = Trainer(**trainer_config, **trainer_kwargs)

    """
    -----   setting learning rate    -----
    """
    if opt.train:
        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if torch.cuda.is_available():
            if lightning_config.trainer.devices == 'auto':
                ngpu = torch.cuda.device_count()
            else:
                ngpu = int(lightning_config.trainer.devices)
        else:
            ngpu = 1

        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            lr = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    lr, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            lr = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {lr:.2e}")

        if config.model.target in ['models.taming.models.vqgan.VQModel',
                                   'models.taming.models.vqgan.VQNoDiscModel',
                                   'models.ldm.models.diffusion.ddpm.LatentDiffusion',
                                   'models.ldm_stable.models.diffusion.ddpm.LatentDiffusion']:
            model.learning_rate = lr
        else:
            model.optimizer_kwargs['lr'] = lr

    """
    -----   training    -----
    """
    if opt.train:
        trainer.fit(model, data)

    """
    -----   testing    -----
    """
    if not opt.train and not trainer.interrupted:
        lightning_config.trainer.devices = 1  # testing only on one device
        evaluation = {}
        for ckpt in os.listdir(ckptdir):
            if ckpt == opt.checkpoint_test or opt.checkpoint_test == "":
                for trial in range(1, opt.trials+1):
                    evaluation[str(trial) + " - " + ckpt] = trainer.test(model, data, ckpt_path=os.path.join(ckptdir, ckpt))[0]
        pd.DataFrame(evaluation).transpose().to_csv(os.path.join(logdir, 'evaluation.csv'))
