# Colorize-Black-White-Images
Colorize Black &amp; White Images
🎨 Black & White Image Colorization GUI
Give old memories a splash of life with AI-powered colorization! This project transforms black and white images into vivid, colorized photographs using deep learning with a simple and elegant Tkinter GUI.


💡 About the Project
This project allows users to:

Upload a black and white image

Apply AI-based colorization

View and compare the original vs colorized image side by side

Built using:

🐍 Python

🧠 OpenCV DNN module

🎨 Pre-trained Caffe model for image colorization

🖼️ Tkinter GUI for a user-friendly interface

🛠️ How It Works
The model used in this project is based on the research paper:

“Colorful Image Colorization”
Richard Zhang, Phillip Isola, Alexei A. Efros
ECCV 2016

The neural network is pre-trained and learns to hallucinate plausible colors for grayscale images using convolutional layers trained on millions of images.

📂 Project Structure
graphql
Copy code
colorize-bw-images/
├── models/
│   ├── colorization_deploy_v2.prototxt      # Model architecture
│   ├── colorization_release_v2.caffemodel   # Pre-trained weights
│   └── pts_in_hull.npy                      # Cluster centers for ab channels
├── logo2.jpg                                # Optional: Logo displayed on GUI launch
├── main.py                                  # GUI code using Tkinter
├── test_script.py                           # Command-line version for debugging
├── result.png                               # Output image after colorization
└── README.md                                # This file
🔧 Requirements
Install the dependencies using pip:

bash
Copy code
pip install numpy opencv-python pillow matplotlib
✅ Ensure you are using Python 3.6 or later.

🚀 Getting Started
GUI Version (main.py)
Place your model files (prototxt, caffemodel, pts_in_hull.npy) inside the models/ folder.

Run the GUI:

bash
Copy code
python main.py
From the File menu, click:

Upload Image to select a black & white image.

Color Image to colorize it using the model.

The colorized image will be saved as result.png and also shown on the right side of the GUI.

CLI Version (test_script.py)
You can test a specific image directly from the command line:

bash
Copy code
python test_script.py
Make sure the script references a valid grayscale image like new.jpg.

🧠 Model Internals
Converts the input B&W image to L*a*b* color space.

Uses the L* channel (lightness) as input to the DNN.

The network predicts the a* and b* channels (color components).

Merges L, a, b channels and converts back to BGR format for viewing.

🖼️ Sample Results
Input (Grayscale)	Output (Colorized)

🧑‍💻 Author
Made with 💙 by Vedanth Rakesh
📧 Email: vedanthrakesh2910@gmail.com

📃 License
This project is released under the MIT License. You are free to use, modify, and distribute with attribution.

🙌 Contributions
Have a better model, UI upgrade, or a cool idea?
Feel free to fork this repo and open a pull request!
Let’s make the past more colorful — together. 🌈
