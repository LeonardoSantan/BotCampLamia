import gym
import random
import numpy as np
from IPython.display import clear_output
from time import sleep

random.seed(1234)
metadata = {"render.modes": ["human", "ansi"]}
streets = gym.make("Taxi-v3")
streets.reset()

initial_state = streets.encode(2, 3, 2, 0)

streets.s = initial_state
streets.P[initial_state]


q_table = np.zeros([streets.observation_space.n, streets.action_space.n])

learning_rate = 0.1
discount_factor = 0.6
exploration = 0.1
epochs = 10000
streets.render()
for tripnum in range(epochs):
    state = streets.reset()
    done = False

    while not done:
        random_value = random.uniform(0, 1)
        if random_value < exploration:
            action = streets.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = streets.step(action)
        clear_output(wait=True)
        prev_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - learning_rate) * prev_q + learning_rate * (
            reward + discount_factor * next_max_q
        )
        q_table[state, action] = new_q
        print("Trip Number" + str(tripnum))
        print(streets.render(mode="ansi"))
        sleep(0.2)
        state = next_state
sleep(2)
q_table[initial_state]

# from IPython.display import clear_output
# from time import sleep

# for tripnum in range(1, 11):
#     state = streets.reset()

#     done = False

#     while not done:
#         action = np.argmax(q_table[state])
#         next_state, reward, done, info = streets.step(action)
#         clear_output(wait=True)
#         print("Trip Number" + str(tripnum))
#         print(streets.render(mode='ansi'))
#         sleep(.5)
#         state = next_state
