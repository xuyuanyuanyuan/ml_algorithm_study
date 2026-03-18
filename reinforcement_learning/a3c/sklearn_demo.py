"""A3C（库近似演示）: 用 A2C 快速展示 Actor-Critic 训练流程。"""


def main():
    # 说明: stable-baselines3 提供 A2C（同步），与 A3C 思想接近。
    try:
        import gymnasium as gym
        from stable_baselines3 import A2C
    except ImportError:
        print("缺少依赖，请安装: pip install stable-baselines3 gymnasium")
        return

    env = gym.make("CartPole-v1")
    model = A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=5, gamma=0.99, verbose=0)
    model.learn(total_timesteps=1000)
    print("=== A3C/A2C Demo ===")
    print("已完成短时训练。")


if __name__ == "__main__":
    main()
