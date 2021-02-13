# continuous control
This project is a solution for continous control problem, where an unity environment is controled by artificial brain.

This environment have 33 box states and 4 box actions.
The states represents position, rotation, velocity, and angular velocities of the arm.
The actions represents joints torque and the values is a number between -1 and 1.
The environment is considered solved if the mean of last 100 episodes is more than or equal 30 points.
The agent will receive +0.1 points if his hand is in the target location.
![](reacher.gif)


A requirements file is disponibilized for install necessary libs, just use pip install -r requirements.txt
download unity agent and paste in navigation directory.

Run Continuous_Control.ipynb on jupyter notebook and follow instructs.

For solve this a reinforcement learning networks was implemented, the hiperameters was seted by comparing score graph.
After train is possible to see an smart agent.




