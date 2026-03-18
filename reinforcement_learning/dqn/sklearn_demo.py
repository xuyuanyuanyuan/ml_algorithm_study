"""DQN（PyTorch 简化演示）: 用神经网络近似 Q 函数。"""


def main():
    # 关键参数: gamma 折扣因子, epsilon 探索率。
    try:
        import gymnasium as gym
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install gymnasium torch")
        return

    env = gym.make("CartPole-v1")
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    gamma, epsilon = 0.95, 0.2

    for _ in range(8):  # 仅做短训练演示
        s, _ = env.reset()
        done = False
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32)
            if torch.rand(1).item() < epsilon:
                a = env.action_space.sample()
            else:
                a = int(torch.argmax(model(s_t)).item())
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ns_t = torch.tensor(ns, dtype=torch.float32)
            q_value = model(s_t)[a]
            with torch.no_grad():
                target = r + (0.0 if done else gamma * torch.max(model(ns_t)).item())
            loss = (q_value - target) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s = ns

    print("=== DQN Demo ===")
    print("已完成极简训练循环。")


if __name__ == "__main__":
    main()
