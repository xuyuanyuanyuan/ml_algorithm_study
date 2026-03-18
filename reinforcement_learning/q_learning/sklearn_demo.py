"""Q-Learning（库环境版）: 在 FrozenLake 上做表格型强化学习。"""


def main():
    # 关键参数: alpha 学习率, gamma 折扣因子, epsilon 探索率。
    try:
        import gymnasium as gym
        import numpy as np
    except ImportError:
        print("缺少依赖，请安装: pip install gymnasium numpy")
        return

    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q = np.zeros((n_states, n_actions))

    alpha, gamma, epsilon = 0.1, 0.95, 0.2
    for _ in range(300):
        state, _ = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            q[state, action] += alpha * (
                reward + gamma * np.max(q[next_state]) - q[state, action]
            )
            state = next_state

    print("=== Q-Learning Demo ===")
    print("训练后 Q 表前两行:")
    print(q[:2])


if __name__ == "__main__":
    main()
