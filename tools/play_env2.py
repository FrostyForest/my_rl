#尝试手动与环境交互
import gymnasium as gym
import time
import numpy as np # 确保导入 numpy

env_id = "CartPole-v1" # 或者你想要测试的其他环境
# env_id = "LunarLander-v2"
# env_id = "ALE/Pong-v5"

try:
    env = gym.make(env_id, render_mode="human")
except TypeError: # 有些环境的 render_mode 可能在创建时不支持，或者默认就是 human
    print(f"无法用 render_mode='human' 创建 {env_id}，尝试默认方式。")
    env = gym.make(env_id)


observation, info = env.reset(seed=42)
terminated = False
truncated = False
total_reward = 0
total_steps = 0

# 确保环境窗口有机会被创建和渲染
if env.render_mode == "human":
    env.render()
    time.sleep(0.1) # 给窗口一点时间

print("--- 环境信息 ---")
print(f"观察空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")
if isinstance(env.action_space, gym.spaces.Discrete):
    print(f"  - 离散动作数量: {env.action_space.n}")
elif isinstance(env.action_space, gym.spaces.Box):
    print(f"  - 连续动作范围: low={env.action_space.low}, high={env.action_space.high}")
print("--------------------")
print(f"\n初始观察 (Step {total_steps}): {observation}")
print("-" * 30)


while not (terminated or truncated):
    # 1. 渲染环境 (如果需要)
    if env.render_mode == "human":
        env.render()

    # 2. 获取用户输入作为动作
    action = None
    while action is None:
        try:
            if isinstance(env.action_space, gym.spaces.Discrete):
                action_input = input(f"输入动作 (0 到 {env.action_space.n - 1}), 或 'q' 退出: ")
                if action_input.lower() == 'q':
                    terminated = True # 用户选择退出
                    break
                action = int(action_input)
                if not env.action_space.contains(action):
                    print("无效的离散动作！请重试。")
                    action = None
            elif isinstance(env.action_space, gym.spaces.Box):
                print(f"输入连续动作 (每个维度用逗号分隔, 例如对于 shape {env.action_space.shape}), 或 'q' 退出: ")
                action_input_str = input("> ")
                if action_input_str.lower() == 'q':
                    terminated = True # 用户选择退出
                    break
                action_parts = [float(x.strip()) for x in action_input_str.split(',')]
                if len(action_parts) == env.action_space.shape[0]:
                    action = np.array(action_parts, dtype=env.action_space.dtype)
                    if not env.action_space.contains(action):
                        print(f"动作超出范围 {env.action_space.low} - {env.action_space.high}！请重试。")
                        action = None
                else:
                    print(f"期望输入 {env.action_space.shape[0]} 个值，但得到了 {len(action_parts)} 个。")
                    action = None
            else:
                print("不支持的动作空间类型进行手动输入。")
                terminated = True # 无法处理，强制退出循环
                break
        except ValueError:
            print("无效输入！请输入数字。")
        except Exception as e:
            print(f"发生错误: {e}")
            action = env.action_space.sample() # 出错时随机选择一个动作
            print(f"已随机选择动作: {action}")

    if terminated : break # 如果在获取输入时决定退出

    # 3. 执行动作
    next_observation, reward, term, trunc, info = env.step(action)
    terminated = term # 更新 terminated 状态
    truncated = trunc # 更新 truncated 状态
    total_reward += reward
    total_steps += 1

    # 4. 打印每个 step 的详细信息
    print(f"\n--- Step {total_steps} ---")
    print(f"  采取的动作: {action}")
    print(f"  下一个观察: {next_observation}")
    print(f"  获得的奖励: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    print(f"  Info: {info}")
    print("-" * 30)

    observation = next_observation # 更新观察以备下一轮

    # 控制游戏速度 (可选)
    time.sleep(0.05) # 可以调整这个值

print("\nEpisode 结束!")
print(f"总步数: {total_steps}")
print(f"总奖励: {total_reward}")

env.close()