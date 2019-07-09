#!usr/bin/env python

import gym
import numpy
import time
from DQN1 import DQN
from gym import wrappers
# ROS packages requir/opt/ros/kinetic/lib/python2.7/dist-packagesed
import sys
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")
sys.path.append("/disk/users_tmp/zwu/wzk/catkin_ws/devel/lib/python2.7/dist-packages")
sys.path.append("/fzi/ids/zwu/wzk/ck_workspace/src/aurora_gazebo/openai_ros/openai_ros/src")
print(sys.path)
import rospy
import rospkg

import tf2_py
import tf2_ros
# import our training environment
from openai_ros.task_envs.shadow_tc import learn_to_pick_ball

import json


if __name__ == '__main__':

    rospy.init_node('shadow_tc_learn_to_pick_ball_qlearn', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('ShadowTcGetBall-v0')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_shadow_tc_openai_example')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/shadow_tc/alpha") # 0.1
    Epsilon = rospy.get_param("/shadow_tc/epsilon") # 0.9
    Gamma = rospy.get_param("/shadow_tc/gamma") # 0.7
    epsilon_discount = rospy.get_param("/shadow_tc/epsilon_discount")
    nepisodes = rospy.get_param("/shadow_tc/nepisodes") # 500
    nsteps = rospy.get_param("/shadow_tc/nsteps") #10000

    # Initialises the algorithm that we are going to use for learning
    qlearn = DQN(N_ACTIONS=env.action_space.n,N_STATES=env.observation_space.shape[0]
                           ) # actions = [0, 1, ... ,7]
    # initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0
    learn_steps = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):  # 500
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False


        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation)) # string process, the problem is state space is too large

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.choose_action(state, learn_steps) 
            rospy.logwarn("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            qlearn.store_transition(o)

            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))
            qlearn.store_transition(state,reward, done, nextState)


            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logwarn("# State in which we will start next step=>" + str(nextState))
            if qlearn.MEMORY_COUNTER > qlearn.MEMORY_CAPACITY:
                qlearn.learn()
                learn_steps += 1



                if not (done):
                    rospy.logwarn("NOT DONE")
                    state = nextState
                else:
                    rospy.logwarn("DONE")
                    break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)

        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))


        f = open('/home/wzk/catkin_ws/src/new_start/my_shadow_tc_openai_example/scripts/q_value.txt','w')
        f.write(str(q_value))
        f.close()
        # jsObj = json.dumps(q_value)
        # fileObject = open('q_value.json', 'w')
        # fileObject.write(jsObj)
        # fileObject.close() 


    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))


    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
