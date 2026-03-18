"""DQN（手写教学简化版）: 经验回放 + 目标网络模板。"""


def main():
    try:
        import random
        from collections import deque
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    # 简化环境: 状态是 4 维随机向量, 动作 0/1, 目标是让动作匹配状态符号。
    def sample_transition():
        s = torch.randn(4)
        a = random.randint(0, 1)
        reward = 1.0 if (s.sum().item() > 0 and a == 1) or (s.sum().item() <= 0 and a == 0) else -1.0
        ns = s + 0.05 * torch.randn(4)
        done = random.random() < 0.1
        return s, a, reward, ns, done

    q_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    target_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    replay = deque(maxlen=500)
    gamma = 0.95

    for step in range(300):
        replay.append(sample_transition())
        if len(replay) < 64:
            continue
        batch = random.sample(replay, 32)
        s = torch.stack([b[0] for b in batch])
        a = torch.tensor([b[1] for b in batch], dtype=torch.long)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        ns = torch.stack([b[3] for b in batch])
        done = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + (1 - done) * gamma * target_net(ns).max(dim=1).values
        loss = ((q - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 60 == 0:
            target_net.load_state_dict(q_net.state_dict())

    print("=== DQN Scratch Template ===")
    print("已演示经验回放与目标网络的核心结构。")


if __name__ == "__main__":
    main()
