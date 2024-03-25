import json
from config.config import Config, DatasetConfig, OptimizerConfig, SchedulerConfig, ModelConfig,LossConfig, MiscConfig

def ParseJsonToObj(configPath):
    with open(configPath) as f:
        parseData = json.load(f)
        result = Config()
        dataset=DatasetConfig()
        optimizer=OptimizerConfig()
        scheduler=SchedulerConfig()
        model=ModelConfig()
        loss=LossConfig()
        misc=MiscConfig()
        dataset.__dict__=parseData["dataset"]
        optimizer.__dict__=parseData["optimizer"]
        scheduler.__dict__=parseData["scheduler"]
        model.__dict__=parseData["model"]
        loss.__dict__=parseData["loss"]
        misc.__dict__=parseData["misc"]
        result.__dict__ = {"dataset":dataset,"optimizer":optimizer,"scheduler":scheduler,"model":model,"loss":loss,"misc":misc}

    return result