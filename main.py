# Import libraries
import gym


# Run CartPole-v0 environment with random action taken each time.
def random_action(count):
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(count):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()


if __name__ == "__main__":
    # run environment with random action
    random_action(1000)
