import gym
from DeepQ import Agent
from utils import plot_learning_curve
import numpy as np

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2,
                  eps_end=0.01, input_dims=[4], lr=0.003)
    scores, eps_history = [],[]
    n_games = 100

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            env.render()
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('ep ', i, 'score ', score,
              'average score ', avg_score,
              'epsilon ', agent.epsilon)

    x = [i+1 for i in range(n_games)]
    filename = 'cartpole.png'
    plot_learning_curve(x, scores, eps_history, filename)
