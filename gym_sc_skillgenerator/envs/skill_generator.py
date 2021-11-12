from typing import Optional, Type

import gym as gym
import numpy as np
import graphviz
from gym import spaces

from gym_sc_skillgenerator.envs.node import EntityTargetTrigger, CooldownCheck, DistanceCheck, ExecutorAnimation, \
    EntityDamageMake, SkillNode, SkillProperties

NODE_CLS: list[Type[SkillNode]] = [EntityTargetTrigger, CooldownCheck, DistanceCheck, EntityDamageMake,
                                   ExecutorAnimation]
NODE_CLS_LEN = len(NODE_CLS)
MAX_NODES_COUNT = 128
MAX_NODE_ARGS = 8


def log_message(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(result)
        return result

    return wrapper


class SkillGeneratorEnv(gym.GoalEnv):
    def __init__(self, expected_props: Optional[SkillProperties] = None) -> None:
        super().__init__()
        self.metadata["render.modes"].append("human")

        self._next_node_id = 0
        self._invalid_actions = 0
        self._root: Optional[SkillNode] = None
        self._expected_props = expected_props
        self._total_nodes = 0
        self._node_queue: list[SkillNode] = []

        self.observation_space = spaces.Dict({
            "observation": spaces.Dict({
                "current_node": spaces.Discrete(NODE_CLS_LEN + 1),
                "selectable_nodes": spaces.multi_binary.MultiBinary(NODE_CLS_LEN),
                "current_node_args": spaces.Box(low=0, high=1, shape=(MAX_NODE_ARGS,)),
                "nodes_count": spaces.Discrete(MAX_NODES_COUNT),
                "siblings_count": spaces.Discrete(MAX_NODES_COUNT),
                "invalid_actions": spaces.Box(low=0, high=np.inf, shape=(1,))
            }),
            "achieved_goal": spaces.Box(low=0, high=float('inf'), shape=(4,)),
            "desired_goal": spaces.Box(low=0, high=float('inf'), shape=(4,))
        })

        self.action_space = spaces.Dict({
            "skip_current": spaces.Discrete(2),
            "new_node": spaces.Discrete(NODE_CLS_LEN),
            "node_args": spaces.Box(low=0, high=1, shape=(MAX_NODE_ARGS,)),
        })

    @staticmethod
    def _get_triggers():
        return np.array([node.get_configure().get("is_trigger") or 0 for node in NODE_CLS], dtype=np.int8)

    def _observe(self):
        ob = {
            "desired_goal": self._expected_props.copy(),
            "observation": {
                "nodes_count": self._total_nodes,
                "invalid_actions": [self._invalid_actions]
            }
        }

        if len(self._node_queue) == 0:
            ob["achieved_goal"] = np.zeros(4)
            ob["observation"]["current_node"] = 0
            ob["observation"]["selectable_nodes"] = self._get_triggers()
            ob["observation"]["current_node_args"] = np.zeros(MAX_NODE_ARGS, dtype=np.int8)
            ob["observation"]["siblings_count"] = 0
        else:
            current_node = self._node_queue[0]

            ob["achieved_goal"] = self._root.get_props().copy()
            ob["observation"]["current_node"] = NODE_CLS.index(self._root.__class__) + 1
            ob["observation"]["selectable_nodes"] = self.get_compatible_nodes()
            ob["observation"]["current_node_args"] = np.resize(np.array(current_node.get_args()), (MAX_NODE_ARGS,))
            ob["observation"]["siblings_count"] = len(current_node.output)
        return ob

    def get_compatible_nodes(self):
        if len(self._node_queue) == 0:
            return self._get_triggers()
        return self._node_queue[0].get_compatible_outputs(NODE_CLS)

    def step(self, action: dict):
        if action["skip_current"] > 0:
            if len(self._node_queue) > 0:
                self._node_queue.pop(0)
            elif self._root is None:
                self._invalid_actions += 1
        else:
            node_cls_id = int(action["new_node"])
            args = np.array(action["node_args"])
            assert 0 <= node_cls_id < NODE_CLS_LEN
            assert args.shape == (MAX_NODE_ARGS,)

            node_cls = NODE_CLS[node_cls_id]

            if self.get_compatible_nodes()[node_cls_id] < 1:
                self._invalid_actions += 1
            else:
                # check node None?
                node = node_cls.from_args(self._next_node_id, args)
                self._next_node_id += 1

                if not self._root:
                    self._root = node
                if len(self._node_queue) > 0:
                    self._node_queue[0].output.append(node)
                self._node_queue.append(node)
                self._total_nodes += 1

        achieved_goal = self._root.get_props().copy() if self._root else np.zeros(4)

        observe = self._observe()
        reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=self._expected_props, info={})
        done = self._total_nodes >= MAX_NODES_COUNT or achieved_goal[0] > self._expected_props[0] or (
                self._root is not None and len(self._node_queue) == 0)
        return observe, reward, done, {}

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        return -np.linalg.norm(achieved_goal - desired_goal) - self._invalid_actions

    def reset(self):
        self._expected_props = SkillProperties.from_array(np.random.rand(4) * 10.0)
        self._next_node_id = 0
        self._root: Optional[SkillNode] = None
        self._total_nodes = 0
        self._invalid_actions = 0
        self._node_queue.clear()

        return self._observe()

    def render(self, mode="human"):
        dot = graphviz.Digraph()

        def traversal(node: SkillNode):
            # print(f"visit {node}")
            dot.node(str(node.id), node.__class__.__name__ + str(node.id))
            for child in node.output:
                traversal(child)
                dot.edge(str(node.id), str(child.id))

        if self._root is not None:
            traversal(self._root)

        dot.view()


class SkillGeneratorWrapper(gym.Wrapper):
    def __init__(self, env: Optional[SkillGeneratorEnv] = None):
        if not env:
            env = SkillGeneratorEnv()
        if not isinstance(env, SkillGeneratorEnv):
            raise TypeError("SkillGeneratorWrapper can be used only for SkillGeneratorEnv")
        super().__init__(env)

        observation_space = {k: v for k, v in env.observation_space.spaces.items() if k != "observation"}
        for k, v in env.observation_space["observation"].spaces.items():
            observation_space[k] = v
        self.observation_space = spaces.Dict(observation_space)

        self.action_space = spaces.Box(low=-1, high=1, shape=(MAX_NODE_ARGS * NODE_CLS_LEN + NODE_CLS_LEN + 1,))

    def step(self, action: np.ndarray):
        new_node_selection = action[1:1 + NODE_CLS_LEN]
        new_node_idx = np.argmax(new_node_selection * self.env.get_compatible_nodes())
        new_node_args = action[1 + NODE_CLS_LEN + new_node_idx * MAX_NODE_ARGS:][:MAX_NODE_ARGS]
        real_action = {
            "skip_current": action[0] > 0,
            "new_node": new_node_idx,
            "node_args": (new_node_args + 1) / 2,
        }
        observation, reward, done, info = super().step(real_action)

        new_observation = self.convert_observation(observation)

        return new_observation, reward, done, info

    def reset(self, **kwargs):
        return self.convert_observation(super().reset(**kwargs))

    @staticmethod
    def convert_observation(observation):
        new_observation = {k: v for k, v in observation.items() if k != "observation"}
        for k, v in observation["observation"].items():
            new_observation[k] = v
        return new_observation
