"""SARSA（手写简化版）: 自定义 GridWorld 在线更新。"""


class GridWorld:
    def __init__(self, size=6):
        self.size = size
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.size - 1, self.state + 1)
        done = self.state == self.size - 1
        reward = 1.0 if done else -0.01
        return self.state, reward, done


def main():
    import numpy as np

    env = GridWorld(size=6)
    q = np.zeros((env.size, 2))
    alpha, gamma, epsilon = 0.2, 0.95, 0.2

    for _ in range(400):
        s = env.reset()
        a = np.random.randint(2) if np.random.rand() < epsilon else int(np.argmax(q[s]))
        done = False
        while not done:
            ns, r, done = env.step(a)
            na = np.random.randint(2) if np.random.rand() < epsilon else int(np.argmax(q[ns]))
            q[s, a] += alpha * (r + gamma * q[ns, na] - q[s, a])
            s, a = ns, na

    print("=== SARSA Scratch ===")
    print(q)


if __name__ == "__main__":
    main()
