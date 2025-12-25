import sys
from typing import Type, Any, Dict
from verl.workers.reward_manager import AbstractRewardManager

from .reward_manager_stepwise import Hamster_Reward_Manager_Step_Wise
from .reward_manager import Hamster_Reward_Manager
from .reward_manager_format import Hamster_Format_Reward_Manager
from .reward_manager_calibration import Hamster_Global_Conf_Reward_Manager
from .reward_manager_calibration_stepwise import Hamster_Step_Conf_Reward_Manager

CLASS_REGISTRY: Dict[str, Type[AbstractRewardManager]] = {
    "Hamster_Reward_Manager": Hamster_Reward_Manager,
    "Hamster_Reward_Manager_Step_Wise": Hamster_Reward_Manager_Step_Wise,
    "Hamster_Format_Reward_Manager" : Hamster_Format_Reward_Manager,
    "Hamster_Global_Conf_Reward_Manager": Hamster_Global_Conf_Reward_Manager,
    "Hamster_Step_Conf_Reward_Manager": Hamster_Step_Conf_Reward_Manager,
}

def build_reward_manager(
    name: str,
    tokenizer,
    **kwargs: Any,
) -> AbstractRewardManager:
    if name not in CLASS_REGISTRY:
        raise KeyError(f"Unknown RewardManager: {name}. Available: {list(CLASS_REGISTRY)}")
    cls = CLASS_REGISTRY[name]
    obj = cls(tokenizer=tokenizer, **kwargs)
    if not isinstance(obj, AbstractRewardManager):
        raise TypeError(f"{name} must inherit from AbstractRewardManager")
    return obj
