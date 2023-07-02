import functools
from ArgReader import *
import os
from Loss import *


def hyper_tune(func):
    """
    Tune hyperparameters of a training function
    HPs to tune: 1.lr; 2.momentum; 3.alpha if LASSO or Ridge loss;

    :param func: training function
    :return:
    """
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        max_cv = float("inf")
        hyper_generator = HyperParamGenerator("./hyper_configs.csv")
        hyper_generator.config_gen()
        train_loader = kwargs["train_loader"]
        val_loader = kwargs["val_loader"]
        base_model = kwargs["model"]
        base_optimizer = kwargs["optimizer"]
        loss_func = kwargs["loss_func"]
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        torch.save(base_optimizer.state_dict(), os.path.join(self.checkpoint_dir, "base_opt.pt"))
        torch.save(base_model.state_dict(), os.path.join(self.checkpoint_dir, "base_mod.pt"))
        for config_ind, config in enumerate(hyper_generator.configs):
            base_model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "base_mod.pt")))
            base_optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "base_opt.pt")))
            print(base_optimizer.param_groups[0]["lr"])
            if "lr" in config.keys():
                for g in base_optimizer.param_groups:
                    g["lr"] = g["lr"] * config["lr"]
            if "momentum" in config.keys():
                for g in base_optimizer.param_groups:
                    g["momentum"] = config["momentum"]
            print(f"Config {config_ind + 1}")
            trained_mod, trained_opt, trained_cv = func(self,
                                                        train_loader=train_loader,
                                                        val_loader=val_loader,
                                                        model=base_model,
                                                        optimizer=base_optimizer,
                                                        loss_func=loss_func)
            if trained_cv < max_cv:
                torch.save((base_model, base_optimizer, trained_cv,
                            base_optimizer.param_groups[0]["lr"], base_optimizer.param_groups[0]["momentum"]),
                           os.path.join(self.checkpoint_dir, "best_hps.pt"))
                max_cv = trained_cv
        best_mod, best_opt, best_cv, best_lr, best_momentum = \
            torch.load(os.path.join(self.checkpoint_dir, "best_hps.pt"))
        print(f"Client {self.client_id} tuned. lr: {best_lr}, momentum: {best_momentum}, cv: {best_cv}")
        return best_mod, best_opt, best_cv
    return wrapped
