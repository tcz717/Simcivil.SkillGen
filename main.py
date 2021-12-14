# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines3 import A2C

from gym_sc_skillgenerator.envs.skill_generator import SkillGeneratorEnv, SkillGeneratorWrapper
from stable_baselines3.common.env_checker import check_env

from azureml.core import Run

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run = Run.get_context()
    env = SkillGeneratorWrapper(SkillGeneratorEnv())

    model = A2C('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)

    model.save("outputs/model")

    obs = env.reset()
    print("training done")
    reward = None
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break

    print("Total steps", i)
    env.render(show=False)
    print(env.observe())
    print(reward)
