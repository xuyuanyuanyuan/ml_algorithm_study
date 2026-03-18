"""A3C（手写教学模板版）: 多 worker 异步更新思想示意。"""


def main():
    try:
        import threading
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    # 全局网络（共享参数）。
    global_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = optim.Adam(global_net.parameters(), lr=1e-2)
    lock = threading.Lock()

    def worker(worker_id):
        # 教学简化环境: 单状态 bandit，动作1回报更高。
        state = torch.tensor([[0.0]])
        for _ in range(120):
            logits = global_net(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            reward = 1.0 if (action.item() == 1 and torch.rand(1).item() > 0.3) else 0.0

            # Actor-Critic 简化: 只做 actor loss，便于初学者看核心结构。
            loss = -dist.log_prob(action) * reward
            with lock:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(f"worker {worker_id} 完成")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    probs = torch.softmax(global_net(torch.tensor([[0.0]])), dim=-1).detach().numpy().ravel()
    print("=== A3C Scratch Template ===")
    print(f"训练后策略概率: action0={probs[0]:.3f}, action1={probs[1]:.3f}")


if __name__ == "__main__":
    main()
