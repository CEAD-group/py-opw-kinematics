# Axis Configuration for ABB

This document describes the axis configuration for the ABB robot.

There are three singularities within the robot’s working range (See RAPID reference manual  RAPID summary, section Motion and I/O principles - Singularities).

* cf1 is the quadrant number for axis 1.
* cf4 is the quadrant number for axis 4.
* cf6 is the quadrant number for axis 6.

|cfx|Wrist center relative to axis 1|Wrist center relative to lower arm|Axis 5 angle|
|---|---|---|---|
|0|In front of|In front of|Positive|
|1|In front of|In front of|Negative|
|2|In front of|Behind|Positive|
|3|In front of|Behind|Negative|
|4|Behind|In front of|Positive|
|5|Behind|In front of|Negative|
|6|Behind|Behind|Positive|
|7|Behind|Behind|Negative|

cfx is used to select one of eight possible robot configurations numbered from 0 through 7.
The table below above describes each one of them in terms of how the robot is positioned relative
to the three singularities.
The pictures below give an example of how the same tool position and orientation is attained
by using the eight different configurations. 
