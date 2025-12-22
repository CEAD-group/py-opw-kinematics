# Robot Visualization Example

This guide explains how to use the included visualization example to see py-opw-kinematics in action. The animation tool helps you understand what the library does by showing robot movement in 3D.

## Purpose

The visualization example serves as:
- **Educational tool**: See how robot kinematics work visually
- **Library demonstration**: Understand what py-opw-kinematics calculates
- **Debugging aid**: Visualize robot poses and trajectories
- **Quick validation**: Check if your robot parameters are working correctly

This is a simple example tool, not a production animation system.

## 📁 What's Included

- `live_robot_animation.py` - Script that demonstrates the library and generates visualization data
- `robot_animation_viewer_collision_shapes.html` - Simple web viewer built with Plotly.js for 3D visualization
- `live_robot_animation.json` - Example output file

## 🚀 Quick Demo

### Step 1: Generate Visualization Data

```bash
# Navigate to examples folder
cd python/examples

# Run the demo (creates live_robot_animation.json)
python3 live_robot_animation.py
```

**What this demonstrates:**
- Creating a robot model with realistic parameters
- Setting up joint constraints (like parallelogram linkages)
- Generating a trajectory using inverse kinematics
- Computing forward kinematics for visualization

**What the JSON file contains:**
The generated `live_robot_animation.json` describes all the data needed for 3D visualization:
- **Joint positions**: Location of each robot joint (J1-J6, base, TCP) for every frame
- **Tool path**: The complete trajectory path that the robot's TCP follows
- **Visualization shapes**: Geometric shapes (spheres, cylinders, boxes) and their colors to represent joints, links, and collision volumes

### Step 2: View the Results

1. **Open** `documentation/robot_animation_viewer_collision_shapes.html` **directly in your browser**
2. **Click "Choose File"** and select the `live_robot_animation.json` file you created

### Step 3: What You'll See

The viewer shows:
- **Robot structure**: Joints, links, and coordinate frames
- **Trajectory path**: The calculated robot movement
- **Real-time animation**: Robot following the computed path
- **Interactive 3D view**: Rotate, zoom, and explore

This helps you understand what the py-opw-kinematics library is calculating behind the scenes.

## 🖥️ Viewing the Animation

### Step 1: Open the Viewer

1. **Open** `documentation/robot_animation_viewer.html` **directly in your browser**
2. **Click "Choose File"** and select the `robot_animation.json` file you created

### Step 2: What You'll See

The 3D viewer displays:

#### **Robot Structure**
- **Joint spheres**: Yellow spheres at each joint location (J1-J6, base, TCP)
- **Link cylinders**: Gray connecting elements between joints  
- **Coordinate frames**: RGB arrows showing X(red), Y(green), Z(blue) axes at key joints
- **TCP frame**: Special coordinate frame at the tool center point

#### **Mechanical Details** (for robots with parallelogram linkage)
- **Parallelogram links**: Cyan/teal colored bars showing the mechanical coupling between J2 and J3
- **Collision shapes**: Semi-transparent protective volumes around robot links
  - Red cylinders around moving parts
  - Teal boxes around major links

#### **Trajectory Visualization**
- **White path line**: Shows the complete TCP trajectory path
- **Moving TCP**: Red sphere following the trajectory path
- **Trajectory buildup**: Path appears progressively as animation plays

### Step 3: Interactive Controls

#### **Animation Controls**
- **Play/Pause button**: Start or stop the animation
- **Speed slider**: Adjust animation speed from 1ms to 2000ms per frame
- **Progress bar**: Click to jump to any frame in the animation  
- **Frame counter**: Shows current frame / total frames


## Limitations

This is a basic visualization example with some limitations:

- **Simple trajectories**: Only generates basic circular paths  
- **Browser based**: Requires a web browser and may be slow for very long animations
- **Educational purpose**: Not intended for production use or complex analysis

---

For more details on the py-opw-kinematics library itself, check the [API Reference](api.md) or try the other [examples](../python/examples/).