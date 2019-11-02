# DQN-Cartpole-Keras-Gym
A Deep Q Learning neural network, created with Keras to balance a pole on a cart in the gym environment.

Three different functions are available:
   1. random_action(count)
   2. train_dqn(episodes, model_name, load_model=False)
   3. show_result(count, model)

### random_action
This function just takes random action and displays how it looks like. The variable cont is set to 100 but, witch means
it takes 100 actions before ending the environment.

### train_dqn
Trains an neural network, which has two inner layers with each 24 neurons. This net is automatically saved after every
10 episodes. The number of episodes can be set with the variable episodes. You also need to pass the model_name under 
which it should be saved / loaded. If the model should be loaded please pass load_model = True.

### show_result
To this function you can give a model and a count (how many actions should be taken), to show you how your model is preforming.

For more checkout the Gym [Docs](http://gym.openai.com/docs/).
