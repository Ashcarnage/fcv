import cv2
import streamlit as st
import numpy as np
from PIL import Image


def segment_color(frame, lower_hue, upper_hue, lower_saturation, upper_saturation, lower_value, upper_value):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for the selected color in HSV
    lower_bound = np.array([lower_hue, lower_saturation, lower_value])
    upper_bound = np.array([upper_hue, upper_saturation, upper_value])

    # Create a mask for the selected color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Bitwise-AND mask and original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result


def main():
    st.title("Live Webcam Color Segmentation")

    # Color sliders for HSV range
    st.sidebar.header("HSV Range Settings")
    lower_hue = st.sidebar.slider("Lower Hue", 0, 180, 0)
    upper_hue = st.sidebar.slider("Upper Hue", 0, 180, 180)
    lower_saturation = st.sidebar.slider("Lower Saturation", 0, 255, 100)
    upper_saturation = st.sidebar.slider("Upper Saturation", 0, 255, 255)
    lower_value = st.sidebar.slider("Lower Value", 0, 255, 100)
    upper_value = st.sidebar.slider("Upper Value", 0, 255, 255)

    # Display the video feed from the webcam
    stframe = st.empty()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        # Perform color segmentation
        segmented_frame = segment_color(frame, lower_hue, upper_hue, lower_saturation, upper_saturation, lower_value,
                                        upper_value)

        # Convert the BGR frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to PIL Image format
        image_pil = Image.fromarray(frame_rgb)

        # Display the frame in the Streamlit app
        stframe.image(image_pil, caption="Color Segmentation", use_column_width=True)

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
