from env.flappy_env import FlappyBirdEnv

env = FlappyBirdEnv()

obs = env.reset()
print("Obs shape:", obs.shape)

obs, reward, done, info = env.step(0)
print("Step output:", reward, done, info)
