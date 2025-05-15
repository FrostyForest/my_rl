#该代码用来手动测试环境
import gymnasium as gym
from gymnasium.utils.play import play

# env_id = "CartPole-v1"
# env_id = "LunarLander-v2"
#env_id = "ALE/Pong-v5" # Atari 游戏效果更好
env_id = "MountainCar-v0" # 连续动作环境

# 创建环境
env = gym.make(env_id, render_mode="rgb_array") # play 通常需要 rgb_array 模式

# 定义按键映射 (可选，play 会尝试提供默认映射)
# 格式: {(key_tuple): action_index_or_value}
# 对于离散动作：
keys_to_action = {
    ("a",): 0, # 假设动作0是向左
    ("s",): 1, # 假设动作1是向右
    ("d",): 2, # 假设动作1是向右
}
# 对于连续动作 (更复杂，通常 play 对连续的默认映射可能不够好):
# 你可能需要一个回调函数来处理按键并生成连续动作值

play(env, keys_to_action=keys_to_action, fps=30, zoom=3)
# 对于很多环境，不指定 keys_to_action，play 会打印出默认的控制方式
#play(env, fps=30, zoom=2)

env.close()
