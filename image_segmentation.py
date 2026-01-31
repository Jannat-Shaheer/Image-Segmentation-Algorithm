import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Step 1: Data Preprocessing
# --------------------------------------------------
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return image, normalized


# --------------------------------------------------
# Step 2 & 3: Image Flattening + K-Means Clustering
# --------------------------------------------------
def apply_kmeans(image, k=3):
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(
        pixel_values,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    labels = labels.flatten()
    segmented = labels.reshape(image.shape)

    tumor_cluster = np.argmax(centers)
    return segmented, tumor_cluster


# --------------------------------------------------
# Step 4: Binary Mask Generation
# --------------------------------------------------
def generate_binary_mask(segmented, tumor_cluster):
    mask = np.zeros_like(segmented, dtype=np.uint8)
    mask[segmented == tumor_cluster] = 255
    return mask


# --------------------------------------------------
# Step 5: Morphological Closing
# --------------------------------------------------
def postprocess_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cleaned


# --------------------------------------------------
# Tumor Localization (Bounding Box)
# --------------------------------------------------
def locate_tumor(mask, original):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return original, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return output, (x, y, w, h)


# --------------------------------------------------
# Visualization
# --------------------------------------------------
def show_results(original, preprocessed, segmented, mask, cleaned, boxed):
    images = [
        original,
        preprocessed,
        segmented * (255 // 2),
        mask,
        cleaned,
        boxed
    ]

    titles = [
        "Original MRI",
        "Preprocessed Image",
        "K-Means Segmentation",
        "Binary Tumor Mask",
        "After Morphological Closing",
        "Tumor Localization"
    ]

    plt.figure(figsize=(14, 9))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main Function
# --------------------------------------------------
def main():
    image_path = "Brain_Image.png"   # replace with your MRI image

    original, preprocessed = preprocess_image(image_path)
    segmented, tumor_cluster = apply_kmeans(preprocessed)
    binary_mask = generate_binary_mask(segmented, tumor_cluster)
    cleaned_mask = postprocess_mask(binary_mask)
    boxed_image, bbox = locate_tumor(cleaned_mask, original)

    show_results(
        original,
        preprocessed,
        segmented,
        binary_mask,
        cleaned_mask,
        boxed_image
    )

    if bbox:
        x, y, w, h = bbox
        print(f"Tumor Location -> X:{x}, Y:{y}, Width:{w}, Height:{h}")
        print("Tumor Area (pixels):", cv2.countNonZero(cleaned_mask))
    else:
        print("No tumor detected.")


if __name__ == "__main__":
    main()
