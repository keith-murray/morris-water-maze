# Morris Water Maze
This repository contains code for my final project for MIT's 9.49 _Neural Circuits for Cognition_ course taught by [Prof. Ila Fiete](https://en.wikipedia.org/wiki/Ila_Fiete). It's a simple deep learning project where I trained a recurrent neural network (RNN) to perform the [morris water maze task](https://en.wikipedia.org/wiki/Morris_water_navigation_task). 

Optimally, the RNN would navigate from its starting location to the goal location along a straight line. Instead, the RNN navigates to a corner and then follows a rehearsed path to the goal location. Despite being suboptimal, the strategy of first navigating to a landmark and then to a desired location is a common navigational strategy. Like a Paris tour guide once told me:
> After a few beers, it's much easier to first walk to the Eiffel Tower and then to your hotel. 

<div align="center">
<img src="https://github.com/keith-murray/morris-water-maze/blob/main/results/simulated_tests_test.gif" alt="morris_water_maze" width="400"></img>
</div>

In the above video, the red square is the RNN agent, the green line is where the RNN is looking, and the yellow square is the goal location.
