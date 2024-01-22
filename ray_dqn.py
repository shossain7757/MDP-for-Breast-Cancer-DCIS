import gym
import gym_breastcancer
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.registry import register_env
import os
from datetime import datetime
import tempfile
from ray.tune.logger import UnifiedLogger
import pickle


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



# Parse Command Line Arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--repeat_training", default=2, type=int, help="Repeatation of Training")
parser.add_argument("-t", "--timesteps", default=50, type=int,help="Training timesteps")
args = vars(parser.parse_args())

## Instantitiate The Environment

env = gym.make('breastcancer-v2')
register_env("breastcancer-v2", lambda config: env)





## Setting the configuration of the network

config = dqn.DEFAULT_CONFIG.copy()
config["model"]["fcnet_hiddens"] = [24,24]
config["model"]["fcnet_activation"] = "relu"
config["dueling"] = True
config["double_q"] = True
config["log_level"] = "ERROR"


## A custom logger function to log the training results in the 
## desired directory

def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator







def train(itr):
    agent = dqn.DQNTrainer(config=config, env="breastcancer-v2",
    logger_creator=custom_log_creator(os.path.expanduser("./dqn_results"), 'dqn_training'))
    for j in range(timesteps):
        agent.train()
    success = []
    for _ in range(0,100):
        reward = 0
        timestep = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.compute_single_action(obs)
            obs, r, done, info = env.step(action)
            reward += r
            timestep += 1
        success.append((reward,timestep))
    with open("agent_performance/dqn{}".format(itr+1), "wb") as fp:
        pickle.dump(success, fp)       
    print('Saved Agent Performance')






    
ray.shutdown()

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True,log_to_driver=False)

    training_time = args['repeat_training']
    timesteps = args['timesteps'] 

    for i in range(training_time):
        train(i)
    print('Finished Training & Evaluation')

    ray.shutdown()



