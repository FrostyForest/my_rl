from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv

model_class = DQN  # 同样适用于 SAC, DDPG 和 TD3
N_BITS = 15

env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

# 可用的策略 (参考论文): future, final, episode
goal_selection_strategy = "future" # 等同于 GoalSelectionStrategy.FUTURE

# 初始化模型
model = model_class(
    "MultiInputPolicy", # HER 通常与处理字典观察的 MultiInputPolicy 一起使用
    env,
    replay_buffer_class=HerReplayBuffer, # 指定使用 HerReplayBuffer
    # HER 的参数
    replay_buffer_kwargs=dict(
        n_sampled_goal=4, # 每个原始转换额外采样的虚拟目标数量
        goal_selection_strategy=goal_selection_strategy, # 目标选择策略
    ),
    verbose=1,
)

# 训练模型
model.learn(1000)

model.save("./her_bit_env")
# 因为需要访问 `env.compute_reward()`
# HER 模型在加载时必须同时加载环境
model = model_class.load("./her_bit_env", env=env)

obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()