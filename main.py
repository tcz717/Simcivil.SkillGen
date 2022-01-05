import sys

from azureml.core import Run
from stable_baselines3 import A2C
from stable_baselines3.common.logger import Logger, HumanOutputFormat

from gym_sc_skillgenerator.envs.logger import AzureRunLogger
from gym_sc_skillgenerator.envs.skill_generator import SkillGeneratorEnv, SkillGeneratorWrapper

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run = Run.get_context()
    is_offline = run.id.startswith("OfflineRun_")
    env = SkillGeneratorWrapper(SkillGeneratorEnv())
    new_logger = Logger(None, output_formats=[HumanOutputFormat(sys.stdout), AzureRunLogger(run)])

    model = A2C('MultiInputPolicy', env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=100000)

    model.save("outputs/model")

    obs = env.reset()
    print("training done")
    reward = None
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward}")
        if done:
            break

    env.render(show=is_offline)
    print(env.observe())
