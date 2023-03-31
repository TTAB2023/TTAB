# -*- coding: utf-8 -*-
from .no_adaptation import NoAdaptation
from .tent import Tent
from .bn_adapt import BNAdapt
from .memo import MEMO
from .shot import SHOT
from .t3a import T3A
from .ttt import TTT
from .ttt_plus_plus import TTTPlusPlus
from .note import NOTE
from .sar import SAR
from .conjugate_pl import ConjugatePL
from .cotta import CoTTA
from .eata import EATA


def get_model_adaptation_method(adaptation_name):
    return {
        "no_adaptation": NoAdaptation,
        "tent": Tent,
        "bn_adapt": BNAdapt,
        "memo": MEMO,
        "shot": SHOT,
        "t3a": T3A,
        "ttt": TTT,
        "ttt_plus_plus": TTTPlusPlus,
        "note": NOTE,
        "sar": SAR,
        "conjugate_pl": ConjugatePL,
        "cotta": CoTTA,
        "eata": EATA,
    }[adaptation_name]
