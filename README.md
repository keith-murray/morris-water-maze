# Morris Water Maze
This repository was my final project for MIT's 9.49 _Neural Circuits for Cognition_ class taught by [Prof. Ila Fiete](https://en.wikipedia.org/wiki/Ila_Fiete). It's a simple deep learning project where I trained a recurrent neural network (RNN) to perform the [morris water maze task](https://en.wikipedia.org/wiki/Morris_water_navigation_task). 

Optimally, the RNN would take a straight line from its location to the goal location. Instead, the RNN usually navigates to a corner and then follows a "rehearsed" path to the goal location. While the solution is suboptimal from an error accumulation perspective, the strategy of first navigating to a landmark and then to the goal location is a common navigational strategy. As Paris tour guide once told me, it's easier, especially after a few beers, to first walk to the Eiffel Tower and then to your hotel instead of navigating directly to your hotel.

<div align="center">
<img src="https://github.com/keith-murray/morris-water-maze/blob/main/results/simulated_tests_test.gif" alt="morris_water_maze" width="400"></img>
</div>

In the above video, the red square is the RNN agent, the green line is where the RNN is looking, and the yellow square is the goal location.
