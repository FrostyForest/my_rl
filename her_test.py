import os
import time
from dataclasses import dataclass, field
import pathlib,random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # 虽然测试时优化器不是必须的，但加载函数可能需要它
import tyro
from torch.distributions.categorical import Categorical # 假设是离散动作

# 假设你的自定义包装器和网络定义在这些文件中
from tools.my_wrapper import MountainCarHerWrapper # 替换为你的包装器路径
# from your_sac_module import Actor, SoftQNetwork, layer_init # 导入你训练时使用的网络定义

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer # 仅为了加载函数完整性

# --- 网络定义 (与训练时保持一致) ---
# 你应该从你训练脚本中复制 Actor 和 SoftQNetwork 的定义到这里
# 或者将它们放在一个共享的模块中导入

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_space_orig = env.observation_space.spaces["observation"]
        obs_dim_orig = np.array(obs_space_orig.shape).prod()
        goal_space = env.observation_space.spaces["desired_goal"]
        goal_dim = np.array(goal_space.shape).prod()
        input_dim = obs_dim_orig + goal_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # 假设是离散动作空间
        self.fc_logits = layer_init(nn.Linear(256, env.action_space.n))

    def forward(self, observation_dict: dict[str, torch.Tensor]):
        obs_features = observation_dict["observation"]
        desired_goal_features = observation_dict["desired_goal"]
        combined_input = torch.cat([obs_features, desired_goal_features], dim=1)
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)
        return logits

    def get_action(self, observation_dict_on_device: dict[str, torch.Tensor], deterministic: bool = False):
        logits = self.forward(observation_dict_on_device)
        policy_dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(policy_dist.probs, dim=-1)
        else:
            action = policy_dist.sample()
        # 测试时通常不需要 log_prob 和 action_probs，除非特定评估指标需要
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1) # 所有动作的log_softmax
        return action, log_prob, action_probs

# SoftQNetwork 定义在测试时不是严格必需的，除非你想评估Q值
# 但加载函数可能期望它来加载状态字典
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_space_orig = env.observation_space.spaces["observation"]
        obs_dim_orig = np.array(obs_space_orig.shape).prod()
        goal_space = env.observation_space.spaces["desired_goal"]
        goal_dim = np.array(goal_space.shape).prod()
        input_dim = obs_dim_orig + goal_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_q = layer_init(nn.Linear(128, env.action_space.n))

    def forward(self, observation_dict: dict[str, torch.Tensor]):
        obs_features = observation_dict["observation"]
        desired_goal_features = observation_dict["desired_goal"]
        combined_input = torch.cat([obs_features, desired_goal_features], dim=1)
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_vals = self.fc_q(x)
        return q_vals


@dataclass
class Args:
    exp_name: str = "sac_her_mountaincar_eval" # 给评估起个名字
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False # 评估时通常希望看到视频
    video_dir: str = "videos_eval" # 评估视频保存目录

    env_id: str = "MountainCar-v0"
    num_eval_episodes: int = 10 # 要运行多少个 episode 进行评估
    render_eval: bool = True # 是否在评估时实时渲染 (如果 capture_video=True，这个可能不需要)

    # --- 加载模型和缓冲区的路径 ---
    # *** 修改为你实际保存的路径 ***
    load_model_path: str = "/home/linhai/code/my_rl/models/MountainCar-v0__her_train__1__1747122878/checkpoint_20000.pt"
    # load_buffer_path: Optional[str] = "models/MountainCar-v0__her_test__1__167xxxxxxx/replay_buffer_100000.pkl" # 测试时通常不需要

    # --- 从加载的模型中恢复的参数 (如果需要，但通常测试脚本会用自己的) ---
    # num_envs: int = 1 # 评估时通常用单个环境


def make_env(env_id, seed, idx, capture_video, run_name, video_dir="videos_eval"): # 添加 video_dir
    def thunk():
        # render_mode="rgb_array" 用于视频录制
        # render_mode="human" 用于实时显示 (如果 args.render_eval 为 True)
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        if args.render_eval and idx == 0 and not (capture_video and idx ==0): # 如果要实时渲染且不录制第一个env
            render_mode = "human"

        env_instance = gym.make(env_id, render_mode=render_mode)
        if capture_video and idx == 0:
            video_save_path = pathlib.Path(video_dir) / run_name
            video_save_path.mkdir(parents=True, exist_ok=True)
            print(f"Recording video to {video_save_path}")
            env_instance = gym.wrappers.RecordVideo(env_instance, str(video_save_path), episode_trigger=lambda x: True) # 录制所有

        env_instance = gym.wrappers.RecordEpisodeStatistics(env_instance) # 仍然需要它来获取 episode 统计
        env_instance = MountainCarHerWrapper(env_instance) # 应用你的 HER 包装器
        env_instance.action_space.seed(seed)
        # env_instance.observation_space.seed(seed) # 也可以给观察空间设种子
        return env_instance
    return thunk

def evaluate_agent(args: Args, actor: Actor, envs: DummyVecEnv, device: torch.device):
    print(f"\nStarting evaluation for {args.num_eval_episodes} episodes...")
    actor.eval() # 将 actor 设置为评估模式 (例如，关闭 dropout)
    all_episode_rewards = []
    all_episode_lengths = []
    all_episode_successes = [] # 假设你的 info 或包装器能提供成功信息

    for episode_num in range(args.num_eval_episodes):
        obs_dict_np = envs.reset() # VecEnv 返回的是批处理的 obs
        dones = np.zeros(envs.num_envs, dtype=bool)
        episode_reward = np.zeros(envs.num_envs)
        episode_length = np.zeros(envs.num_envs, dtype=int)
        is_successful_episode = np.zeros(envs.num_envs, dtype=bool)
        action_np=0
        step=0
        while not (dones[0]): # 假设评估时只关注第一个环境 (n_envs=1)
            # 1. 将 NumPy 字典观察转换为 PyTorch 张量字典并移动到设备
            obs_tensor_dict = {}
            for key, value in obs_dict_np.items():
                # 对于评估，我们通常只有一个环境，所以value的第一个维度是1 (或者没有)
                # 需要确保形状是 (1, ...feature_dims...)
                tensor_val = torch.as_tensor(value, dtype=torch.float32, device=device)
                if tensor_val.ndim == len(envs.observation_space.spaces[key].shape): # 如果缺少批次维度
                    tensor_val = tensor_val.unsqueeze(0)
                obs_tensor_dict[key] = tensor_val


            # 2. Actor 选择动作 (确定性)
            with torch.no_grad():
                if step%6==0:
                    action_tensor, log_prob, action_probs = actor.get_action(obs_tensor_dict, deterministic=False)
            action_np = action_tensor.cpu().numpy()
            print(action_probs)

            # 3. 环境执行动作
            next_obs_dict_np, rewards_np, dones_, infos = envs.step(action_np)

            # 假设我们只评估第一个环境 (envs.num_envs 通常为 1 用于评估)
            obs_dict_np = next_obs_dict_np
            episode_reward[0] += rewards_np[0]
            episode_length[0] += 1
            dones=dones_

            if "is_success" in infos[0]: # 假设你的环境或包装器在 info 中提供了成功标志
                is_successful_episode[0] = infos[0]["is_success"]

            if args.render_eval and not args.capture_video and envs.num_envs == 1: # 仅在单个环境且不录制时实时渲染
                 envs.render() # 对于 VecEnv，render 可能需要特殊处理或不可用
                               # 通常直接在单个环境实例上调用 render
            step=step+1

        all_episode_rewards.append(episode_reward[0])
        all_episode_lengths.append(episode_length[0])
        all_episode_successes.append(is_successful_episode[0])
        print(f"Episode {episode_num + 1}: Reward={episode_reward[0]:.2f}, Length={episode_length[0]}, Success={is_successful_episode[0]}")

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_length = np.mean(all_episode_lengths)
    success_rate = np.mean(all_episode_successes) * 100

    print("\n--- Evaluation Summary ---")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print("--- -------------------- ---")
    actor.train() # 将 actor 恢复到训练模式


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # --- 创建环境 (通常评估时 num_envs=1) ---
    # 为了评估，我们通常只需要一个环境实例
    eval_env_fns = [make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args.video_dir)]
    eval_envs = DummyVecEnv(eval_env_fns)


    # --- 初始化网络 (结构必须与保存时一致) ---
    actor = Actor(eval_envs).to(device)
    # qf1 = SoftQNetwork(eval_envs).to(device) # 测试时主要需要 Actor
    # qf2 = SoftQNetwork(eval_envs).to(device)
    # qf1_target = SoftQNetwork(eval_envs).to(device)
    # qf2_target = SoftQNetwork(eval_envs).to(device)

    # --- 加载模型 ---
    if args.load_model_path and os.path.exists(args.load_model_path):
        print(f"Loading model checkpoint from {args.load_model_path}")
        checkpoint = torch.load(args.load_model_path, map_location=device)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        # qf1.load_state_dict(checkpoint["qf1_state_dict"]) # 如果需要加载 Q 网络
        # qf2.load_state_dict(checkpoint["qf2_state_dict"])
        # qf1_target.load_state_dict(checkpoint["qf1_target_state_dict"])
        # qf2_target.load_state_dict(checkpoint["qf2_target_state_dict"])

        loaded_args = checkpoint.get("args")
        if loaded_args:
            print(f"Model was trained with args: {loaded_args.env_id}, total_timesteps: {loaded_args.total_timesteps}")
        print("Model loaded successfully.")
    else:
        print(f"Error: Model checkpoint not found at {args.load_model_path}")
        exit()

    # --- 进行评估 ---
    evaluate_agent(args, actor, eval_envs, device)

    eval_envs.close()
    print("Evaluation finished.")