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
        self._done = False
        self._node_used = {k: 0 for k in range(NODE_CLS_LEN)}

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

    def observe(self):
        ob = {
            "desired_goal": self._expected_props.copy(),
            "achieved_goal": self._root.get_props().copy() if self._root else np.zeros(4),
            "observation": {
                "nodes_count": self._total_nodes,
                "invalid_actions": [self._invalid_actions],
                "selectable_nodes": self.get_compatible_nodes()
            }
        }

        if len(self._node_queue) == 0:
            ob["observation"]["current_node"] = 0
            ob["observation"]["current_node_args"] = np.zeros(MAX_NODE_ARGS, dtype=np.int8)
            ob["observation"]["siblings_count"] = 0
        else:
            current_node = self._node_queue[0]

            ob["observation"]["current_node"] = NODE_CLS.index(current_node.__class__) + 1
            ob["observation"]["current_node_args"] = np.resize(np.array(current_node.get_args()), (MAX_NODE_ARGS,))
            ob["observation"]["siblings_count"] = len(current_node.output)
        return ob

    def get_compatible_nodes(self):
        if len(self._node_queue) == 0 and self._root:
            return np.zeros(NODE_CLS_LEN)
        if len(self._node_queue) == 0:
            return self._get_triggers()
        return self._node_queue[0].get_compatible_outputs(NODE_CLS)

    def step(self, action: dict):
        node_num_exceed_max_type = False
        new_node_type = False
        self._last_invalid_action = False
        if action["skip_current"] > 0:
            if len(self._node_queue) > 0:
                self._node_queue.pop(0)
            elif self._root is None:
                self._invalid_actions += 1
                self._last_invalid_action = True
        else:
            node_cls_id = int(action["new_node"])
            args = np.array(action["node_args"])
            assert 0 <= node_cls_id < NODE_CLS_LEN
            assert args.shape == (MAX_NODE_ARGS,)

            node_cls = NODE_CLS[node_cls_id]

            if self.get_compatible_nodes()[node_cls_id] < 1:
                self._invalid_actions += 1
                self._last_invalid_action = True
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
                new_node_type = self._node_used[node_cls_id] == 0
                self._node_used[node_cls_id] += 1
                node_num_exceed_max_type = self._node_used[node_cls_id] > 1

        achieved_goal = self._root.get_props().copy() if self._root else np.zeros(4)

        info = {
            "node_num_exceed_max_type": node_num_exceed_max_type,
            "new_node_type": new_node_type
        }
        observe = self.observe()
        done = self._total_nodes >= MAX_NODES_COUNT or (achieved_goal > self._expected_props * 1.1).any() or (
                self._root is not None and len(self._node_queue) == 0)
        self._done = done
        reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=self._expected_props, info=info)
        return observe, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        reward = 0
        if self._done:
            goal_diff = achieved_goal - desired_goal
            if achieved_goal.sum() <= 1:  # or (achieved_goal > 1.5 * desired_goal).any():
                return -1e7
            goal_diff *= np.array([1000, 800, 800, 800])
            reward += -np.linalg.norm(goal_diff)

        if self._last_invalid_action:
            reward += -1000.0

        if info["node_num_exceed_max_type"]:
            reward += -1000.0

        if info["new_node_type"]:
            reward += 100.0

        if len(self._node_queue) > 0 and len(self._node_queue[0].output) > 0:
            reward += -100.0

        return reward

    def reset(self):
        self._expected_props = SkillProperties.from_array(np.random.rand(4) * [4, 10, 10, 10] + 1.0)
        self._next_node_id = 0
        self._root: Optional[SkillNode] = None
        self._total_nodes = 0
        self._invalid_actions = 0
        self._done = False
        self._node_queue.clear()
        self._node_used = {k: 0 for k in range(NODE_CLS_LEN)}

        return self.observe()

    def render(self, mode="human", show=True):
        dot = graphviz.Digraph()

        def traversal(node: SkillNode):
            # print(f"visit {node}")
            dot.node(str(node.id), node.__class__.__name__ + str(node.id), xlabel=node.xlable())
            for child in node.output:
                traversal(child)
                dot.edge(str(node.id), str(child.id))

        if self._root is not None:
            traversal(self._root)

        print(show)
        if show:
            dot.view()
        else:
            dot.save(directory="outputs")


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
        new_node_selection = action[1:1 + NODE_CLS_LEN] + 2  # ensure all are above 0
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

    def observe(self):
        return self.env.observe()
