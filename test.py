import gymnasium as gym

# 创建 CartPole 环境的实例
# v1 是环境版本号，推荐指定版本以保证可复现性
env = gym.make("FrozenLake-v1", render_mode="human") # render_mode="human" 用于可视化
# 如果不需要可视化，可以不设置 render_mode 或设为 "rgb_array" 获取图像数据
# env = gym.make("CartPole-v1")

# 查看动作空间 (Action Space)
print("Action Space:", env.action_space)
# 对于 CartPole，是 Discrete(2)，表示有两个离散动作 (0:向左推, 1:向右推)
# 可以从中采样随机动作
random_action = env.action_space.sample()
print("Sample random action:", random_action)

# 查看观察空间 (Observation Space)
print("Observation Space:", env.observation_space)
# 对于 CartPole，是 Box([-4.8 -inf -0.418 -inf], [4.8 inf 0.418 inf], (4,), float32)
# 表示一个包含 4 个连续值的向量 (车位置, 车速度, 杆角度, 杆角速度)
# Box 类型表示连续空间，包含每个维度的最小值和最大值



import gymnasium as gym
import time # 用于暂停观察

# 1. 创建环境
# render_mode="human" 会弹出一个窗口显示环境状态
# 如果在服务器或不需要可视化运行，可以去掉 render_mode 或设为 "rgb_array"
env = gym.make("FrozenLake-v1", render_mode="human")

# --- 运行多个回合 (Episode) ---
num_episodes = 10000

for episode in range(num_episodes):
    # 2. 重置环境到初始状态，获取第一个观察值
    # reset() 返回 observation 和 info 字典
    observation, info = env.reset(seed=42) # 设置 seed 使结果可复现
    print(f"\n--- Episode {episode+1} Start ---")
    print("Initial Observation:", observation)

    terminated = False
    truncated = False
    total_reward = 0
    steps = 0

    # --- 单个回合的循环 ---
    while not terminated and not truncated:
        # 3. (可选) 渲染环境
        # 在 render_mode="human" 时，这会更新弹出的窗口
        # 如果设为 "rgb_array"，它会返回一个 numpy 数组表示的图像
        env.render()

        # 4. 选择动作 (这里我们先用随机动作)
        # 在实际 RL 中，这里应该是你的智能体根据 observation 决定 action
        action = env.action_space.sample()
        # print(f"Step {steps+1}: Taking action {action}")

        # 5. 执行动作，获取环境反馈
        # step() 返回: 新观察, 奖励, 是否终止, 是否截断, 信息字典
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation)

        # 累积奖励
        total_reward += reward
        steps += 1

        # (可选) 稍微暂停一下，方便人类观察
        # time.sleep(0.02)

        # 检查回合是否结束
        if terminated:
            print(f"Episode finished naturally after {steps} steps (Terminated).")
        if truncated:
            print(f"Episode finished due to time limit after {steps} steps (Truncated).")

    print(f"Episode {episode+1} finished. Total Reward: {total_reward}")
    # --- 单个回合结束 ---

# --- 所有回合结束 ---

# 6. 关闭环境，释放资源
env.close()
print("\nEnvironment closed.")