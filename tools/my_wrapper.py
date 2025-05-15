import gymnasium as gym
from gymnasium import spaces
from gymnasium import ObservationWrapper, RewardWrapper # RewardWrapper 可能是关键
import numpy as np

# 假设这是我们想要修改的原始环境 (它可能没有字典观察空间或 compute_reward)
# class OriginalEnv(gym.Env):
#     def __init__(self):
#         super().__init__()
#         self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 原始观察是2D位置
#         self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
#         self.goal_position = np.array([0.5, 0.5])
#         self.current_position = np.zeros(2)
#         self.distance_threshold = 0.1

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_position = np.random.uniform(-0.5, 0.5, size=(2,))
#         return self.current_position.astype(np.float32), {}

#     def step(self, action):
#         self.current_position += action * 0.1 # 简化移动
#         self.current_position = np.clip(self.current_position, -1, 1)

#         distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
#         terminated = distance_to_goal < self.distance_threshold
#         reward = 0.0 if terminated else -1.0 # 稀疏奖励
#         truncated = False # 假设没有时间限制

#         return self.current_position.astype(np.float32), reward, terminated, truncated, {}

#     def render(self): pass
#     def close(self): pass


# 现在创建包装器
class MountainCarHerWrapper(gym.ObservationWrapper, gym.RewardWrapper, gym.ActionWrapper):
    def __init__(self, env, goal_sampling_fn=None, distance_threshold=0.05):
        super().__init__(env) # 初始化 ObservationWrapper (会自动调用 env.reset() 获取原始 obs space)
        # ActionWrapper 和 RewardWrapper 不需要特别的 super().__init__ 参数

        # 原始环境的观察空间 (例如，智能体自身状态)
        self.original_observation_space = env.observation_space

        # 假设 goal 的维度与原始观察空间的前 N 维相同，或者是一个独立的空间
        # 这里简化，假设 goal 空间与原始观察空间相同
        # 你需要根据你的环境具体定义 goal_space
        self.goal_space = spaces.Box(
            low=np.array([env.observation_space.low[0]]), # 或具体定义 goal 的范围
            high=np.array([env.observation_space.high[0]]),
            shape=(1,), # 或 goal 的特定形状
            dtype=env.observation_space.dtype,
        )

        # 定义新的字典观察空间
        self.observation_space = spaces.Dict(
            {
                "observation": self.original_observation_space,
                "achieved_goal": self.goal_space, # 通常与 'observation' 中的一部分对应
                "desired_goal": self.goal_space,
            }
        )
        # 动作空间保持不变 (除非你也想包装它)
        # self.action_space = env.action_space

        self.goal_sampling_fn = goal_sampling_fn if goal_sampling_fn else self._default_sample_goal
        self.distance_threshold = distance_threshold
        self.desired_goal = None # 在 reset 时设置

    def _get_achieved_goal(self, observation):
        # 这个函数需要从原始的 'observation' 中提取或计算出 'achieved_goal'
        # 这是一个非常重要的部分，需要根据你的环境具体实现
        # 例如，如果原始观察就是位置，achieved_goal 也是位置
        return np.array([observation[0]], dtype=self.observation_space.dtype) # 简化示例：假设原始观察就是可达成的目标

    def _default_sample_goal(self):
        # 一个默认的目标采样函数
        # 你应该根据你的任务需求来实现更复杂的目标采样
        #goal=self.goal_space.sample()
        goal= np.array([0.5], dtype=np.float32)
        return goal

    def reset(self, **kwargs):
        # 重置原始环境
        original_obs, info = self.env.reset(**kwargs)
        self.desired_goal = self.goal_sampling_fn()
        return self.observation(original_obs), info

    def observation(self, observation):
        # 将原始环境的观察转换为字典格式
        return {
            "observation": observation,
            "achieved_goal": self._get_achieved_goal(observation),
            "desired_goal": self.desired_goal.copy(), # 确保每次都是副本
        }

    def step(self, action):
        # 执行原始环境的步骤
        original_next_obs, original_reward, terminated, truncated, info = self.env.step(action)

        # 将原始 next_obs 转换为字典格式
        dict_next_obs = self.observation(original_next_obs)

        # 使用新的 compute_reward (通过 RewardWrapper 实现)
        # 注意：这里的 reward 方法是 RewardWrapper 提供的，它会调用我们定义的 self.reward()
        reward = self.reward(original_reward, terminated, truncated, info, achieved_goal=dict_next_obs["achieved_goal"], desired_goal=self.desired_goal)

        # 检查是否因为达到了当前 desired_goal 而成功 (如果需要修改 terminated 状态)
        # 如果原始环境的 terminated 条件与 HER 的成功条件不同，你可能需要在这里调整
        # 例如，即使原始环境没有终止，但达到了 HER 的目标，也应该视为成功
        is_her_success = self._is_success(dict_next_obs["achieved_goal"], self.desired_goal)
        if is_her_success and not terminated: # 如果 HER 成功但原始环境未终止
             terminated = True # 对于 HER 来说，这个子目标达成了
             # reward = 0.0 # 也可以在这里覆盖奖励

        # 如果 episode 结束 (terminated or truncated)，为下一个 reset 准备新的 desired_goal
        # if terminated or truncated:
        #     # 注意：新的 desired_goal 会在下一次 reset 时才生效于观察中
        #     # 这里的 info 可以用来传递给 HerReplayBuffer，如果需要的话
        #     if "episode" not in info: # 确保有 'episode' 字典
        #          info["episode"] = {}
        #     # HerReplayBuffer 通常需要 info 来计算奖励，但我们直接在包装器中计算了
        #     # 如果需要，可以将 self.desired_goal 放入 info，但这不标准
        #     pass


        return dict_next_obs, reward, terminated, truncated, info

    # 这是 RewardWrapper 需要我们实现的方法
    def reward(self, reward, terminated, truncated, info, achieved_goal, desired_goal):
        # 这个方法会被 self.step() 中调用 RewardWrapper 的 reward 方法时调用
        # 参数 reward, terminated, truncated, info 是从原始 env.step() 来的
        # 我们在这里实现 compute_reward 的逻辑
        return self.compute_reward(achieved_goal, desired_goal, info)


    # 这是 HER 需要的核心方法
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """
        计算基于当前达到的目标和期望目标的奖励。
        :param achieved_goal: 当前达到的目标
        :param desired_goal: 期望达到的目标
        :param info: 额外信息字典
        :return: 计算得到的奖励
        """
        # 如果输入是单个样本 (goal_dim,)，将其扩展为 (1, goal_dim) 以便统一处理
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        #  N 是批处理大小
        N = desired_goal.shape[0]
        rewards_arr = np.zeros(N, dtype=np.float32) # 初始化奖励数组

        # 计算距离，distance 的形状会是 (N,)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # 按元素应用你的条件逻辑来计算 target_reward
        # 我们需要为每个样本计算 target_reward
        # desired_goal[:, 0] 表示取所有样本的 desired_goal 的第一个维度（假设目标位置是其第一个维度）
        # 或者如果 desired_goal 本身就是那个标量值，你需要确保它被正确广播或索引

        # 假设 desired_goal 的形状是 (N, goal_dim) 并且我们关心的是 goal_dim 中的某个特定索引，比如第0个
        # 或者如果 goal_dim 就是 1 (例如 MountainCar 的位置)
        goal_values_to_check = desired_goal[:, 0] # 取每个 desired_goal 的第一个元素

        # 条件1: desired_goal < -0.6
        cond1 = goal_values_to_check < -0.6
        rewards_arr[cond1] = 0.0

        # 条件2: -0.6 <= desired_goal < 0.5
        cond2 = np.logical_and(goal_values_to_check >= -0.6, goal_values_to_check < 0.5)
        # 对于满足 cond2 的元素，计算 target_reward
        # (desired_goal[cond2, 0] + 0.6) / 1.25
        rewards_arr[cond2] = (goal_values_to_check[cond2] + 0.6) / 1.25

        # 条件3: desired_goal >= 0.5
        cond3 = goal_values_to_check >= 0.5
        rewards_arr[cond3] = 1.5
        
        # 现在 rewards_arr 包含了根据 desired_goal 条件计算的 target_reward
        # 最后，根据 distance 应用这个 target_reward 或 0
        final_rewards = np.where(distance < self.distance_threshold, rewards_arr, 0.0)
        return final_rewards.astype(np.float32)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """
        判断是否成功达到期望目标。
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return distance < self.distance_threshold

# --- 用于 ActionWrapper 的示例方法 (如果需要修改动作) ---
#    def action(self, action):
#        # 如果需要，在这里修改动作
#        return action

#    def reverse_action(self, action):
#        # 如果需要，在这里反向修改动作 (不常用)
#        return action