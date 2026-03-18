"""PPO（库版本）: 使用 stable-baselines3 进行快速演示。"""


def main():
    # 关键参数: n_steps 每次采样步数, clip_range 裁剪系数。
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
    except ImportError:
        print("缺少依赖，请安装: pip install stable-baselines3 gymnasium")
        return

    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=64, learning_rate=3e-4, verbose=0)
    model.learn(total_timesteps=1000)
    print("=== PPO Demo ===")
    print("已完成短时训练。")


if __name__ == "__main__":
    main()
