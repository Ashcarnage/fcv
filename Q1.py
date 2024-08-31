import cv2
import streamlit as st
import numpy as np
from PIL import Image

def canny_edge_detection(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

def hough_transform(image, threshold):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny_edge_detection(image)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=50, maxLineGap=10)

    return lines, edges

def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    image = np.array(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def main():
    st.title("Hough Transform and Canny Edge Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        option = st.selectbox("Select Transform", ["Canny Edge Detection", "Hough Transform"])

        if option == "Canny Edge Detection":
            edges = canny_edge_detection(image)
            st.image(edges, caption="Canny Edge Detection", use_column_width=True)

        elif option == "Hough Transform":
            threshold = st.slider("Threshold", min_value=50, max_value=200, step=10, value=100)
            lines, edges = hough_transform(image, threshold)
            image_with_lines = draw_lines(image, lines)

            # Display the Canny edges and Hough Transform results
            st.image(edges, caption="Canny Edge Detection", use_column_width=True)
            st.image(image_with_lines, caption="Hough Transform", use_column_width=True)

if __name__ == "__main__":
    main()


