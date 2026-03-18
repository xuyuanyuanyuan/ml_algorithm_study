"""PPO（手写教学简化版）: 用 bandit 演示 clipped objective。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    # 极简 bandit: 动作1期望奖励更高。
    def sample_reward(action):
        return 1.0 if (action == 1 and torch.rand(1).item() > 0.3) else 0.0

    policy = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 2))
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    clip_eps = 0.2

    state = torch.tensor([[0.0]])
    for _ in range(200):
        logits = policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        old_log_prob = dist.log_prob(action).detach()
        reward = torch.tensor([sample_reward(int(action.item()))], dtype=torch.float32)

        # PPO 核心: ratio * advantage 的裁剪目标。
        new_logits = policy(state)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        new_log_prob = new_dist.log_prob(action)
        ratio = torch.exp(new_log_prob - old_log_prob)
        advantage = reward  # 教学简化: 不估计价值函数，直接用奖励近似。
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
        loss = -torch.min(surr1, surr2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probs = torch.softmax(policy(state), dim=-1).detach().numpy().ravel()
    print("=== PPO Scratch Template ===")
    print(f"最终策略概率: action0={probs[0]:.3f}, action1={probs[1]:.3f}")


if __name__ == "__main__":
    main()
