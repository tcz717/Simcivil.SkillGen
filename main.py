# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines3 import A2C

from gym_sc_skillgenerator.envs.skill_generator import SkillGeneratorEnv, SkillGeneratorWrapper
from stable_baselines3.common.env_checker import check_env

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = SkillGeneratorWrapper(SkillGeneratorEnv())

    # check_env(env)
    # check_env(SkillGeneratorWrapper(env), skip_render_check=False)

    model = A2C('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    print("training done")
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break

    print(i)
    env.render()
    obs = env.reset()
    print(obs)
