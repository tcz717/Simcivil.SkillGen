from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from enum import IntFlag
from typing import Type, TypedDict

import numpy as np

MIN_SKILL_RANGE = 99.0

MAX_SKILL_RANGE = 100.0

MAX_COOLDOWN = 60 * 60 * 24 * 3


class SkillDataType(IntFlag):
    Entity = 1,
    EntityList = 1 << 1,
    Position = 1 << 2,
    Range = 1 << 3,


class SkillProperties(np.ndarray):

    @staticmethod
    def from_array(array):
        props = SkillProperties(shape=(4,))
        props[:] = array[:]
        return props

    @staticmethod
    def zero():
        props = SkillProperties(shape=(4,), buffer=np.zeros(4))
        return props

    @property
    def complexity(self):
        return self[0]

    @property
    def power(self):
        return self[1]

    @property
    def difficulty(self):
        return self[2]

    @property
    def fanciness(self):
        return self[3]

    @difficulty.setter
    def difficulty(self, value):
        self[2] = value

    @complexity.setter
    def complexity(self, value):
        self[0] = value

    @power.setter
    def power(self, value):
        self[1] = value

    @fanciness.setter
    def fanciness(self, value):
        self[3] = value


class SkillConfiguration(TypedDict):
    name: str
    is_trigger: bool
    forbid_leaf: bool
    allowed_input: SkillDataType
    allowed_output: SkillDataType


class SkillNode(ABC, metaclass=ABCMeta):
    def __init__(self, node_id: int, *args):
        self.id = node_id
        self.output = []
        self._args = args

    def get_props(self) -> SkillProperties:
        return self.sum_children_props()

    def sum_children_props(self):
        if len(self.output) == 0:
            return SkillProperties.zero()
        return SkillProperties.from_array(np.sum([node.get_props() for node in self.output], axis=0))

    def get_compatible_outputs(self, node_cls: list[Type[SkillNode]]):
        return np.array([(not node.get_configure().get("is_trigger", False)) and
                         self.get_configure().get("allowed_output") ^ node.get_configure().get("allowed_output")
                         for node in node_cls], dtype=np.int8)

    def get_args(self):
        return self._args

    @classmethod
    def from_args(cls, node_id: int, args):
        node = cls.__new__(cls)
        cls.check_normalized(args)
        cls.__init__(node, node_id, *args)
        return node

    @staticmethod
    def check_normalized(args):
        arr = np.array(args)
        if (arr < 0).any() or (arr > 1).any():
            raise ValueError("Args must be normalized")

    @classmethod
    @abstractmethod
    def get_configure(cls) -> SkillConfiguration:
        pass


class EntityTargetTrigger(SkillNode):

    @classmethod
    def get_configure(cls):
        return {
            'name': cls.__name__,
            'is_trigger': True,
            'forbid_leaf': False,
            'allowed_input': SkillDataType.Entity,
            'allowed_output': SkillDataType.Entity
        }


class CooldownCheck(SkillNode):

    def __init__(self, node_id: int, cooldown=1.0, *_):
        super().__init__(node_id, cooldown)
        # Max cooldown 3 days
        self.cooldown = cooldown * MAX_COOLDOWN

    @classmethod
    def get_configure(cls):
        return {
            'name': cls.__name__,
            'forbid_leaf': False,
            'allowed_input': SkillDataType.Entity,
            'allowed_output': SkillDataType.Entity
        }

    def get_props(self):
        props = super(CooldownCheck, self).get_props()
        props.difficulty += self.cooldown / 5 + 1
        return props


class DistanceCheck(SkillNode):
    @classmethod
    def from_args(cls, node_id: int, args):
        args = np.array(args[:2])
        args.sort()
        return DistanceCheck(node_id, *args)

    def __init__(self, node_id: int, max_range: float, min_range=0, *_):
        super().__init__(node_id, max_range, min_range)
        self.min_range = min_range * MIN_SKILL_RANGE
        self.max_range = max_range * MAX_SKILL_RANGE

    @classmethod
    def get_configure(cls):
        return {
            'name': cls.__name__,
            'is_trigger': False,
            'forbid_leaf': False,
            'allowed_input': SkillDataType.Entity,
            'allowed_output': SkillDataType.Entity
        }

    def get_props(self):
        props = super(DistanceCheck, self).get_props()
        props.difficulty += self.min_range * 2 + 10 / (self.max_range + 1)
        return props


class EntityDamageMake(SkillNode):
    def __init__(self, node_id: int, base_damage: float, accuracy: float, damage_type: int, *_):
        super().__init__(node_id, base_damage, accuracy, damage_type)
        self.base_damage = base_damage * 1000
        self.accuracy = accuracy
        self.damage_type = int(damage_type * 10)

    @classmethod
    def get_configure(cls):
        return {
            'name': cls.__name__,
            'is_trigger': False,
            'forbid_leaf': False,
            'allowed_input': SkillDataType.Entity,
            'allowed_output': SkillDataType.Entity
        }

    def get_props(self):
        props = super(EntityDamageMake, self).get_props()
        props.power += (self.base_damage * 2 + self.accuracy * 5) + self.damage_type
        return props


class ExecutorAnimation(SkillNode):
    def __init__(self, node_id: int, animation_idx=0, *_):
        super().__init__(node_id, int(animation_idx))
        self.animation_idx = int(animation_idx * 10)

    @classmethod
    def get_configure(cls):
        return {
            'name': cls.__name__,
            'is_trigger': False,
            'forbid_leaf': False,
            'allowed_input': SkillDataType.Entity,
            'allowed_output': SkillDataType.Entity
        }

    def get_props(self):
        props = super(ExecutorAnimation, self).get_props()
        props.fanciness += int(self.animation_idx)
        return props
