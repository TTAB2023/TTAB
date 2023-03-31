# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here, e.g., balancing hyper-parameters
#    in the objective function of SHOT.
# 3. We provide default hyperparameters if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "no_adaptation": {"model_selection_method": "last_iterate"},
    "tent": {
        "model_selection_method": "last_iterate",
        # This hyperparameter can be used to assign pretraining task. More specifically,
        # if auxiliary_loaders=None, it means the TTA method does not need to adjust the pretraining pipeline,
        # or it directly loads pretrained model parameters without re-pretraining using auxiliary_loaders.
        "optimizer": "SGD",
    },
    "bn_adapt": {
        "model_selection_method": "last_iterate",
        "optimizer": "SGD",
        "adapt_prior": 0,  # the ratio of training set statistics.
    },
    "memo": {
        "prior_psz": -1,
        "aug_size": 32,
        "optimizer": "SGD",
        "bn_prior_strength": 16,
    },
    "t3a": {"top_M": 100},
    "shot": {
        "offline_nepoch": 30,
        "optimizer": "SGD", # Adam for officehome
        "auxiliary_batch_size": 32,
        "threshold_shot": 0.9,  # confidence threshold for online shot.
        "ent_par": 1.0,
        "cls_par": 0.3,  # 0.1 for officehome.
    },
    "ttt": {
        "entry_of_shared_layers": "layer2",
        "optimizer": "SGD",
        "aug_size": 32,
        "threshold_ttt": 1.0,
        "dim_out": 4,  # For rotation prediction self-supervision task.
        "rotation_type": "rand",
    },
    "ttt_plus_plus": {
        "entry_of_shared_layers": None,
        "optimizer": "SGD",
        "batch_size_align": 256,
        "queue_size": 256,
        "offline_nepoch": 500,
        "bnepoch": 2,  # first few epochs to update bn stat.
        "delayepoch": 0,  # In first few epochs after bnepoch, we dont do both ssl and align (only ssl actually).
        "stopepoch": 25,
        "scale_ext": 0.5,
        "scale_ssh": 0.2,
        "align_ext": True,
        "align_ssh": True,
        "fix_ssh": False,
        "method": "align",  # choices = ['ssl', 'align', 'both']
        "divergence": "all",  # choices = ['all', 'coral', 'mmd']
    },
    "note": {
        "optimizer": "SGD",  # use Adam in the paper
        "memory_size": 64,
        "update_every_x": 64,  # This param may change in our codebase.
        "memory_type": "PBRS",
        "bn_momentum": 0.01, 
        "temperature": 1.0,
        "loss_scalar": 0.0,
        "iabn": False,  # replace bn with iabn layer
        "iabn_k": 4,
        "threshold_note": 1,  # skip threshold to discard adjustment.
        "use_learned_stats": True,
    },
    "sar": {
        "sar_margin_e0": math.log(1000)*0.40, # The threshold for reliable minimization in SAR.
        "reset_constant_em": 0.2, # threshold e_m for model recovery scheme
        "optimizer": "SGD",
    },
    "conjugate_pl": {
        "temperature_scaling": 1.0,
        "model_eps": 0.0, # this should be added for Polyloss model.
    },
    "cotta": {
        "optimizer": "Adam",
        "alpha_teacher": 0.999, # weight of moving average for updating the teacher model.
        "restore_prob": 0.01, # the probability of restoring model parameters.
        "threshold_cotta": 0.92, # Threshold choice discussed in supplementary 
    },
    "eata": {
        "eata_margin_e0": math.log(1000)*0.40, # The threshold for reliable minimization in EATA.
        "eata_margin_d0": 0.05, # for filtering redundant samples.
        "fisher_size": 2000, # number of samples to compute fisher information matrix.
        "fisher_alpha": 50, # the trade-off between entropy and regularization loss.
        "optimizer": "SGD",
    },
}
