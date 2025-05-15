import gymnasium as gym
import numpy as np
import time
#用随机动作来测试环境
def test_environment_with_random_actions(env_id: str, num_episodes: int = 5, max_steps_per_episode: int = 200, render_mode: str = None, seed: int = None):
    """
    使用随机动作测试一个 Gymnasium 环境。

    参数:
    env_id (str): 要创建的环境的 ID (例如 "CartPole-v1", "MountainCar-v0")。
    num_episodes (int): 要运行的总回合数。
    max_steps_per_episode (int): 每个回合的最大步数。
    render_mode (str, optional): 渲染模式 (例如 "human", "rgb_array")。默认为 None (不渲染)。
    seed (int, optional): 环境的随机种子。默认为 None。
    """

    print(f"正在测试环境: {env_id}")

    # 创建环境
    # 使用 try-except 来处理可能的渲染模式问题或环境创建问题
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except Exception as e:
        print(f"创建环境 {env_id} 时发生错误 (render_mode='{render_mode}'): {e}")
        print(f"尝试不使用 render_mode 创建环境...")
        try:
            env = gym.make(env_id)
            print(f"成功创建环境 {env_id} (无渲染)。")
        except Exception as e_no_render:
            print(f"创建环境 {env_id} (无渲染) 也失败了: {e_no_render}")
            return

    # 设置随机种子 (如果提供)
    # 注意：对于环境本身的随机性，通常在 reset 时设置种子更有效
    # env.action_space.seed(seed) # 设置动作空间的种子（如果需要的话，但通常 reset 时的种子更重要）

    print(f"观察空间 (Observation Space): {env.observation_space}")
    print(f"动作空间 (Action Space): {env.action_space}")
    print(f"  - 动作空间类型: {type(env.action_space)}")
    if isinstance(env.action_space, gym.spaces.Discrete):
        print(f"  - 离散动作数量: {env.action_space.n}")
    elif isinstance(env.action_space, gym.spaces.Box):
        print(f"  - 连续动作形状: {env.action_space.shape}")
        print(f"  - 动作下限: {env.action_space.low}")
        print(f"  - 动作上限: {env.action_space.high}")

    total_steps_taken = 0
    for episode in range(num_episodes):
        print(f"\n--- 开始回合 {episode + 1}/{num_episodes} ---")
        # 重置环境，获取初始观察和 info
        # 在每个 episode 开始时设置种子，可以确保可复现性（如果环境支持）
        current_seed = seed + episode if seed is not None else None
        observation, info = env.reset(seed=current_seed)
        print(f"初始观察 (形状 {observation.shape if isinstance(observation, np.ndarray) else type(observation)}): {observation}")

        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0

        start_time=time.time()
        for step in range(max_steps_per_episode):
            # 1. 随机选择一个动作
            if step%100==99:
                end_time=time.time()
                sps = step / (end_time-start_time)
                print(f"平均 SPS (Steps Per Second): {sps:.2f}")
            if step%5==0:
                action = env.action_space.sample()

            # 2. 执行动作
            next_observation, reward, terminated, truncated, info = env.step(action)

            # 3. 渲染环境 (如果指定)
            if render_mode == "human":
                env.render()
                time.sleep(0.01) # 加入少量延迟以便观察

            # 4. 打印信息
            # print(f"  步骤: {step + 1}")
            # print(f"    动作: {action}")
            # print(f"    奖励: {reward}")
            # print(f"    下一个观察 (形状 {next_observation.shape if isinstance(next_observation, np.ndarray) else type(next_observation)}): {next_observation}")
            # print(f"    是否终止 (Terminated): {terminated}")
            # print(f"    是否截断 (Truncated): {truncated}")
            # if info: # 打印不为空的 info 字典
            #     print(f"    信息 (Info): {info}")

            episode_reward += reward
            episode_steps += 1
            total_steps_taken += 1

            # 更新观察
            observation = next_observation

            # 检查 episode 是否结束
            if terminated or truncated:
                break

        print(f"--- 回合 {episode + 1} 结束 ---")
        print(f"  总奖励: {episode_reward}")
        print(f"  总步数: {episode_steps}")

    print(f"\n--- 测试完成 ---")
    print(f"总运行步数: {total_steps_taken}")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    # --- 选择一个环境进行测试 ---

    # 示例 1: 经典控制 - CartPole (离散动作)
    # test_environment_with_random_actions("CartPole-v1", num_episodes=3, max_steps_per_episode=100, render_mode="human", seed=42)

    # 示例 2: 经典控制 - MountainCar (离散动作)
    # test_environment_with_random_actions("MountainCar-v0", num_episodes=2, max_steps_per_episode=200, render_mode="human", seed=10)

    # 示例 3: Box2D - LunarLander (离散动作)
    # 需要安装 pip install gymnasium[box2d]
    # test_environment_with_random_actions("LunarLander-v2", num_episodes=2, max_steps_per_episode=300, render_mode="human", seed=1)

    # 示例 4: MuJoCo - Hopper (连续动作)
    # 需要安装 MuJoCo 和 pip install gymnasium[mujoco]
    # test_environment_with_random_actions("Hopper-v4", num_episodes=2, max_steps_per_episode=100, render_mode="human", seed=1)

    # 示例 5: Atari 环境 (离散动作)
    # 需要安装 pip install gymnasium[atari,accept-rom-license] ale-py
    # test_environment_with_random_actions("PongNoFrameskip-v4", num_episodes=1, max_steps_per_episode=200, render_mode="human", seed=5)

    # 示例 6: 测试一个不存在或有问题的环境ID
    # test_environment_with_random_actions("NonExistentEnv-v0")

    # ---- 在这里选择你想要测试的环境 ----
    # test_environment_with_random_actions("你的环境ID", num_episodes=3, max_steps_per_episode=50, render_mode="human")
    # 例如，如果你想测试你的 HER 兼容环境，确保它已经被正确注册，或者你能直接实例化它。
    # 如果是 HER 兼容环境，它的 observation 会是字典，动作可能是连续的。
    # from your_her_env_setup import make_goal_conditioned_env # 假设你有这个
    # def thunk():
    #     return make_goal_conditioned_env("CartPole-v1") # 包装一个简单环境为例
    # test_env_instance = thunk() # 创建单个实例来获取信息
    # print(f"HER Env Observation Space: {test_env_instance.observation_space}")
    # print(f"HER Env Action Space: {test_env_instance.action_space}")
    # test_env_instance.close()
    # # 注意：直接将 thunk 或包装后的单个环境传递给这里可能不直接工作，
    # # 因为 test_environment_with_random_actions 期望一个 env_id 来 gym.make()
    # # 你可以修改函数以接受一个已创建的环境实例，或者确保你的自定义环境已注册。

    # 让我们测试一个已知的连续动作环境
    try:
        test_environment_with_random_actions("MountainCar-v0", num_episodes=100, max_steps_per_episode=200, render_mode="human", seed=7)
    except Exception as e:
        print(f"运行 Pendulum-v1 示例时出错: {e}")
        print("请确保已安装相关依赖，例如：pip install gymnasium[classic_control]")