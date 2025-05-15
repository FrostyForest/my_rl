import gymnasium as gym
from gymnasium import spaces # 导入 spaces 模块用于类型检查
import numpy as np
# 创建一个环境实例 (选择一个你熟悉或想了解的环境)
# env_id = "CartPole-v1"          # Discrete Box (低维向量)
# env_id = "PongNoFrameskip-v4"   # Box (图像)
env_id = "MountainCar-v0"         # Dict (假设你有一个基于目标的字典观察空间环境)
# 替换为你自己的已注册环境ID (如果你有自定义环境)
try:
    # 假设 MyGoalEnv-v0 的注册和定义如下 (简化版)
    if env_id == "MyGoalEnv-v0" and env_id not in gym.envs.registry:
        from gymnasium.envs.registration import register
        register(
            id="MyGoalEnv-v0",
            entry_point="your_module:MyGoalEnvClass", # 替换为你的环境类路径
        )
        # 你需要确保 your_module:MyGoalEnvClass 是可导入的
        # 为了这个例子能运行，我们定义一个简单的 MyGoalEnvClass
        class MyGoalEnvClass(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Dict({
                    "observation": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                    "achieved_goal": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    "desired_goal": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                })
                self.action_space = spaces.Discrete(2)
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                obs = {
                    "observation": self.observation_space["observation"].sample(),
                    "achieved_goal": self.observation_space["achieved_goal"].sample(),
                    "desired_goal": self.observation_space["desired_goal"].sample(),
                }
                return obs, {}
            def step(self, action):
                obs = {
                    "observation": self.observation_space["observation"].sample(),
                    "achieved_goal": self.observation_space["achieved_goal"].sample(),
                    "desired_goal": self.observation_space["desired_goal"].sample(), # 目标通常不变
                }
                return obs, 0, False, False, {}
            def render(self): pass
            def close(self): pass
        # 注意：上面的 MyGoalEnvClass 仅为示例，你需要有完整的实现
        # 并且 'your_module' 需要替换成实际的模块名，或者如果在本文件定义，则为 '__main__'
        # 为了简单起见，我们换回一个标准环境
        env_id = "CartPole-v1"


    env = gym.make(env_id)
    obs_space = env.observation_space

    print(f"--- 环境: {env_id} ---")
    print(f"观察空间对象类型: {type(obs_space)}")
    print(f"观察空间repr: {obs_space}") # 直接打印对象，会显示其类型和主要参数

    # 2. 通用属性 (所有 Space 类型都有)
    print(f"\n--- 通用属性 ---")
    print(f"形状 (Shape): {obs_space.shape}") # 对于非结构化空间 (如 Box, Discrete)
    print(f"数据类型 (dtype): {obs_space.dtype}") # 对于非结构化空间

    # 3. 根据空间类型获取特定信息

    # a. 如果是 Box 空间 (连续值，可能有界或无界)
    if isinstance(obs_space, spaces.Box):
        print(f"\n--- Box 空间特定信息 ---")
        print(f"下限 (Low): {obs_space.low}")
        print(f"上限 (High): {obs_space.high}")
        print(f"是否有界 (Bounded below): {obs_space.is_bounded('below')}")
        print(f"是否有界 (Bounded above): {obs_space.is_bounded('above')}")
        print(f"是否完全有界 (Is bounded): {obs_space.is_bounded()}") # 等价于上面两个都为True

    # b. 如果是 Discrete 空间 (离散整数值)
    elif isinstance(obs_space, spaces.Discrete):
        print(f"\n--- Discrete 空间特定信息 ---")
        print(f"离散值的数量 (n): {obs_space.n}")
        print(f"起始值 (start, Gymnasium 0.2 Discrete): {getattr(obs_space, 'start', 'N/A')}") # Gymnasium 0.2 引入了 start

    # c. 如果是 Dict 空间 (字典，包含其他 Space 对象)
    elif isinstance(obs_space, spaces.Dict):
        print(f"\n--- Dict 空间特定信息 ---")
        print(f"包含的子空间 (Spaces):")
        for key, subspace in obs_space.spaces.items():
            print(f"  键 '{key}':")
            print(f"    类型: {type(subspace)}")
            print(f"    形状: {subspace.shape}")
            print(f"    数据类型: {subspace.dtype}")
            if isinstance(subspace, spaces.Box):
                print(f"    Box 下限: {subspace.low}")
                print(f"    Box 上限: {subspace.high}")
            elif isinstance(subspace, spaces.Discrete):
                print(f"    Discrete 数量 (n): {subspace.n}")
            # 可以为其他类型的子空间添加更多检查

    # d. 如果是 Tuple 空间 (元组，包含其他 Space 对象)
    elif isinstance(obs_space, spaces.Tuple):
        print(f"\n--- Tuple 空间特定信息 ---")
        print(f"包含的子空间 (Spaces):")
        for idx, subspace in enumerate(obs_space.spaces):
            print(f"  索引 {idx}:")
            print(f"    类型: {type(subspace)}")
            print(f"    形状: {subspace.shape}")
            print(f"    数据类型: {subspace.dtype}")
            # ... 类似 Dict 的进一步检查 ...

    # e. 其他空间类型 (MultiBinary, MultiDiscrete)
    elif isinstance(obs_space, spaces.MultiBinary):
        print(f"\n--- MultiBinary 空间特定信息 ---")
        print(f"形状/向量长度 (n): {obs_space.n}") # n 可以是整数或元组

    elif isinstance(obs_space, spaces.MultiDiscrete):
        print(f"\n--- MultiDiscrete 空间特定信息 ---")
        print(f"每个离散维度的上限 (nvec): {obs_space.nvec}") # 每个维度可以有不同的离散值数量

    # 4. 采样一个观察 (有助于理解结构)
    print(f"\n--- 采样一个观察 ---")
    sample_obs = obs_space.sample()
    print(f"采样观察的类型: {type(sample_obs)}")
    print(f"采样观察的值: {sample_obs}")
    if isinstance(sample_obs, np.ndarray):
        print(f"采样观察的形状 (NumPy): {sample_obs.shape}")
        print(f"采样观察的数据类型 (NumPy): {sample_obs.dtype}")
    elif isinstance(sample_obs, dict):
        print(f"采样观察的键 (Dict): {sample_obs.keys()}")
        for key, value in sample_obs.items():
            if isinstance(value, np.ndarray):
                print(f"  键 '{key}': 形状={value.shape}, 数据类型={value.dtype}")


    env.close()

except ImportError:
    print(f"请确保你已经定义了 'your_module:MyGoalEnvClass' 或者将 env_id 改为 Gymnasium 内置的环境 ID。")
except gym.error.Error as e:
    print(f"创建或检查环境 {env_id} 时出错: {e}")