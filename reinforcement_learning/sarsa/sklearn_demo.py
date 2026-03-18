"""SARSA（库环境版）: 在线更新的时序差分控制。"""


def main():
    try:
        import gymnasium as gym
        import numpy as np
    except ImportError:
        print("缺少依赖，请安装: pip install gymnasium numpy")
        return

    env = gym.make("FrozenLake-v1", is_slippery=False)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    alpha, gamma, epsilon = 0.1, 0.95, 0.2

    for _ in range(300):
        s, _ = env.reset()
        a = env.action_space.sample() if np.random.rand() < epsilon else int(np.argmax(q[s]))
        done = False
        while not done:
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            na = env.action_space.sample() if np.random.rand() < epsilon else int(np.argmax(q[ns]))
            q[s, a] += alpha * (r + gamma * q[ns, na] - q[s, a])
            s, a = ns, na

    print("=== SARSA Demo ===")
    print(q[:2])


if __name__ == "__main__":
    main()
