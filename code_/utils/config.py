from yacs.config import CfgNode as CfgNode

_C = CfgNode()
cfg = _C

# ==================== official variable ====================
_C.LOGGER_PATH = 'outputs'
_C.GPUID = [1]

# ----- Scenario of continual learning ------
_C.DOMAIN_INCR = False

# ------------ Agent ------------------------
_C.AGENT = CfgNode()
# _C.AGENT.TYPE = 'trainer'
# _C.AGENT.NAME = 'CLTrainer'
_C.AGENT.TYPE = 'worker'
_C.AGENT.NAME = 'CLWorker'

_C.AGENT.MODEL_TYPE = 'resnet'
_C.AGENT.MODEL_NAME = 'resnet50'

_C.AGENT.REG_COEF = 0.1
_C.AGENT.FIX_BN = False
_C.AGENT.FIX_HEAD = True # only for domain incremental

# ---------- Dataset --------------------------
_C.DATASET = CfgNode()
_C.DATASET.ROOT = ''
_C.DATASET.NAME = ''
_C.DATASET.NUM_CLASSES = 200
_C.DATASET.NUM_TASKS = 25

# ==================== customer variable ====================
_C.SEED = 0
_C.PRINT_FREQ = 100

# ---------- Dataset --------------------------
_C.DATASET.BATCHSIZE = 64
_C.DATASET.NUM_WORKERS = 4

# --------- Optimizer --------------------------
_C.OPT = CfgNode()
_C.OPT.NAME = 'SGD'
_C.OPT.LR = 0.01
_C.OPT.MOMENTUM = 0.9
_C.OPT.WEIGHT_DECAY = 5e-4
_C.OPT.SCHEDULE = [20, 2]
_C.OPT.GAMMA = 0.1




def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    return config