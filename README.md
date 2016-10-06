# Baxter robot OpenAI Gym Environment

This package integrates the Gazebo simulator with the OpenAI Gym. The environments require the
[baxter interface](http://sdk.rethinkrobotics.com/wiki/Baxter_Interface) to be installed. For the
most part, the setup transfers to the physical robot, but the **BaxterAvoiderEnv** requires joint
states that are retrieved from a Gazebo topic.

# Environments

Before running any environment you should start Gazebo and initialize the Baxter connection.

## BaxterReacherEnv

Simply reach a given location in coordinate space.

## BaxterAvoiderEnv

Reach a given location in coordinate space while avoiding passing though a certain area.
