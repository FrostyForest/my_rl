# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MountainCar-v0"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 8
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod(),#计算 NumPy 数组中所有元素的乘积。这有效地将多维观察空间的形状“展平”为一个单一的数值，代表了观察向量中元素总数。
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_q = layer_init(nn.Linear(128, envs.single_action_space.n))

    def forward(self, x):
        #x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_vals = self.fc_q(x)
        return q_vals


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_logits = layer_init(nn.Linear(256, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)#通过将 logits 传递给 Categorical 的构造函数，我们创建了一个或一批（如果 batch_size > 1）分类分布对象。
        action = policy_dist.sample()#这个方法会根据由 logits 定义的概率分布随机采样一个动作。
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    #assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.AdamW(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    autoreset = np.zeros(envs.num_envs)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "episode" in infos:
        #     for info in infos["final_info"]:
        #         if info is not None:
        #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #             break
              
        if "_episode" in infos:
            # 获取回报、长度和时间的数组
            returns = infos["episode"]["r"]
            lengths = infos["episode"]["l"]
            times = infos["episode"]["t"]
            # 获取哪些是有效的结束标志
            finished_mask = infos["_episode"] # 或者 infos["episode"]["_r"] 等

            for i in np.where(finished_mask)[0]: # 遍历所有结束了的子环境的索引
                print(f"Env {i}: Episodic Return={returns[i]}, Length={lengths[i]}, Time={times[i]}")
                writer.add_scalar(f"charts/episodic_return_env{i}", returns[i], global_step)
                writer.add_scalar(f"charts/episodic_length_env{i}", lengths[i], global_step)

            
                    
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        empty_idx=[]
        for idx, trunc in enumerate(truncations):
            if not autoreset[idx]:
                empty_idx.append(idx)
            else:
                if len(empty_idx)>0:
                    new_idx=random.choice(empty_idx)
                    real_next_obs[idx] = real_next_obs[new_idx]
                    obs[idx] = obs[new_idx]
                    actions[idx]=actions[new_idx]
                    rewards[idx]=rewards[new_idx]
                    terminations[idx]=terminations[new_idx]
                    infos[idx]=infos[new_idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        autoreset = np.logical_or(terminations, truncations)
        # for idx, trunc in enumerate(truncations):#调试
        #     if autoreset[idx]:
        #         print(infos)
        #         breakpoint()
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations)
                qf2_next_target = qf2_target(data.next_observations)
                min_qf_next_target = next_state_action_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                )
                min_qf_next_target = min_qf_next_target.sum(dim=1)#现在是一个形状为 (batch_size,) 的张量，其中每个元素是对应下一个状态的目标价值v'(s')
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

            qf1_values = qf1(data.observations)
            qf2_values = qf2(data.observations)
            qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
            qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()#value model参数每一步都更新

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                #actor model参数延迟更新，args.policy_frequency 控制 Actor（策略）和 Alpha（如果 autotune 开启）的更新频率。例如，如果 args.policy_frequency = 2，那么 Actor 和 Alpha 每隔 2 次 Critic 更新才会进行一次更新。
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    _, log_pi, action_probs = actor.get_action(data.observations)
                    with torch.no_grad():
                        qf1_pi = qf1(data.observations)
                        qf2_pi = qf2(data.observations)
                        min_qf_values = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:#Alpha (熵温度参数) 自动调整
                        alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()#Alpha 的损失函数旨在驱动平均熵接近target_entropy

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
