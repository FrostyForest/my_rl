import gymnasium as gym
import torch
import numpy as np
import time
# 确保你的 Agent 类定义可用 (可以从训练脚本中复制或导入)
from ppo import Agent, make_env, Args # 假设你的训练脚本名为 your_training_script.py
import random

def evaluate_agent(model_path, env_id, num_episodes=10, seed=42, device="cpu"):
    """
    加载并评估一个训练好的 Agent 模型。

    Args:
        model_path (str): 已保存模型 state_dict 的路径。
        env_id (str): 要评估的环境 ID (应与训练时一致)。
        num_episodes (int): 要运行的评估回合数。
        seed (int): 环境和 PyTorch 的随机种子。
        device (str): 运行模型的设备 ('cpu' 或 'cuda')。
    """
    run_name = f"eval_{int(time.time())}" # 为可能的视频录制创建一个运行名称

    # 1. 创建环境 (通常只需要一个实例进行评估)
    #    注意: make_env 可能需要调整，确保它适用于评估场景
    #    例如，你可能只想录制第一个回合的视频，或者完全不录制
    #    这里的 capture_video=True 且 idx=0 是为了演示，你可以按需修改

    eval_env = gym.vector.SyncVectorEnv(
         [lambda: make_env(env_id, 0, capture_video=False, run_name=run_name)()]
         # 如果不需要并行，或者想更简单地处理单个环境，可以只用 gym.make:
         # eval_env = make_env(env_id, 0, capture_video=True, run_name=run_name)()
         # 如果用了 gym.make，后续与环境交互的方式会略有不同 (不需要索引 [0])
    )
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed) # 如果你的 make_env 或 Agent 用了 random

    # 2. 实例化 Agent 模型结构
    #    这里的 envs 参数需要与模型定义时匹配，即使只用一个环境，
    #    Agent 类内部可能还是依赖 SyncVectorEnv 的属性。
    #    或者，你可以修改 Agent 类使其更灵活。
    agent = Agent(eval_env).to(device)

    # 3. 加载保存好的模型权重
    #    map_location=device 确保权重加载到正确的设备
    agent.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    # 4. 设置模型为评估模式
    #    这很重要，会禁用 Dropout 和 Batch Normalization 的更新
    agent.eval()

    total_rewards = []
    total_lengths = []

    print(f"Starting evaluation for {num_episodes} episodes...")

    for episode in range(num_episodes):
        episode_reward = 0
        episode_length = 0
        # 注意：因为用了 SyncVectorEnv，即使只有一个环境，reset 和 step 也是批处理的
        # 如果你改用了 gym.make 创建单个环境，这里就不需要 [0] 索引
        next_obs, _ = eval_env.reset(seed=seed + episode) # 为每个回合设置不同种子（可选）
        next_obs = torch.Tensor(next_obs).to(device)
        terminated = truncated = False

        while not (terminated or truncated):
            # 在评估模式下，不需要计算梯度
            with torch.no_grad():
                # 通常评估时我们想要确定性动作，但 PPO 的 get_action_and_value 默认是采样的
                # 如果想获得最可能的动作，需要修改 get_action_and_value 或直接从 logits 计算
                # 这里我们还是用原方法，但注意这是带随机性的评估
                action, _, _, _ = agent.get_action_and_value(next_obs)

            # 与环境交互 (注意 action 需要转回 CPU 和 NumPy)
            action_ = action.cpu().numpy().reshape(1, -1).squeeze(-1) # 适配 SyncVectorEnv
            next_obs, reward, terminations, truncations, infos = eval_env.step(action_)

            # 处理返回值 (因为是向量环境，需要索引)
            terminated = terminations[0]
            truncated = truncations[0]
            episode_reward += reward[0]
            episode_length += 1
            next_obs = torch.Tensor(next_obs).to(device) # 更新观测

            # (可选) 渲染环境，如果环境支持且 render_mode="human"
            # eval_env.render() # 对于 SyncVectorEnv，渲染可能需要特殊处理或不支持

            # 检查是否有回合结束的信息 (来自 RecordEpisodeStatistics 包装器)
            if "final_info" in infos:
                 final_info = infos["final_info"][0] # 获取第一个环境的信息
                 if final_info is not None and "episode" in final_info:
                     print(f"Episode {episode + 1}: Length={final_info['episode']['l']}, Return={final_info['episode']['r']:.2f}")
                     total_lengths.append(final_info['episode']['l'])
                     total_rewards.append(final_info['episode']['r'])
                     # 因为 SyncVectorEnv 会自动重置，这里理论上不需要 break，
                     # 但如果用了单个 env.make，则回合结束后需要 break
                     # break


    eval_env.close()

    # 打印评估总结
    if total_rewards:
        print("\nEvaluation Summary:")
        print(f"Average Return: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
        print(f"Average Length: {np.mean(total_lengths):.2f} +/- {np.std(total_lengths):.2f}")
    else:
        print("No complete episodes recorded during evaluation.")


if __name__ == "__main__":
    # --- 评估配置 ---
    SAVED_MODEL_PATH = "/home/linhai/code/my_rl/runs/FrozenLake-v1__ppo__1__1746200261/ppo_iter401.pt" # <--- 修改为你实际的模型路径!
    ENV_ID = "FrozenLake-v1" # <--- 确保与训练时一致
    NUM_EVAL_EPISODES = 5
    EVAL_SEED = 123
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_agent(SAVED_MODEL_PATH, ENV_ID, NUM_EVAL_EPISODES, EVAL_SEED, DEVICE)