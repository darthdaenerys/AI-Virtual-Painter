# AI Virtual Painter 🎨

This is an application that enables one to vitually paint in the air using their fingers. It is developed in python on openCV and Mediapipe. So go ahead and recreate your imaginations in the air !

![Alt text](paint.gif)

## Getting Started

- Use the command to clone this repository to your machine
`git clone https://github.com/darthdaenerys/Virtual-Painter`

- Now change current directory to the folder `cd Virtual-Painter`

- `pip install -r requirements.txt`

- `python paint.py`

## Features

- Can draw on your System screen based on your Index finger movement
- Can track your hand in real-time
- Change colour for your brush
- Change your brush size
- Save your beautiful creations by pressing `S`.
- Clear for a fresh screen by pressing `C`.
- Train for your own use case from scratch.

## Working

- This project is a use case of Hand Tracking technology.
- As soon as the user shows up his hand in the camera the application detects it.
- The left hand determines the `standby` or `drawing` mode.
- To Select different color or eraser from the top of Canvas, User must select it by taking his Index finger at the top of icon. The same can also be selected using mouse click.

## Note

Feel free to file a new issue with a respective title and description on the AI-Virtual-Paint. If you already found a solution to your problem, I would love to review your pull request!
