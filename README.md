# Document Analysis and Recognition

![image](https://github.com/vinitshetty16/-Document-Analysis-and-Recognition/assets/63487624/35d0b87b-585e-4b0c-b685-a11a6020556d)

This repository documents the process and outcomes of document analysis and recognition tasks, particularly focusing on deskewing document images and improving text recognition using OCR techniques.

## Introduction

Document analysis and recognition are crucial components of many information processing systems, enabling the extraction of valuable insights and data from textual documents. This project explores various methodologies and techniques to enhance document analysis and recognition, with a specific focus on deskewing document images to improve OCR accuracy.

## Methodology

The methodology involves multiple tasks:

1. **Task 1**: Generating negative images with candidate points selected based on specific strategies.
2. **Task 2**: Experimenting with different candidate point strategies and density thresholds to estimate skew angles.
3. **Task 3**: Deskewing images using the most effective strategy identified in Task 2 and assessing the results.
4. **Task 4**: Performing text recognition (OCR) on both skewed and deskewed document images and comparing the outcomes.

## Dependencies

- Tesseract-OCR
- Python Imaging Library (PIL)
- NumPy
- OpenCV
- Other necessary Python libraries for image processing and text recognition

## Results and Insights

- Different candidate point selection strategies and density thresholds affect skew angle estimation and deskewing accuracy.
- Deskewed document images significantly improve OCR performance compared to skewed images.
- Editable PDFs generated from deskewed document images retain text recognition, enabling easy editing and manipulation of document content.

## Conclusion

The document analysis and recognition tasks demonstrate the effectiveness of deskewing techniques in improving OCR performance. By identifying optimal candidate point strategies and density thresholds, we can accurately estimate skew angles and generate high-quality deskewed images. Subsequently, performing OCR on these deskewed images leads to better text recognition results compared to skewed images. The ability to convert deskewed document images into editable PDFs further enhances the utility of this approach in document processing workflows.

## Future Work

- Explore advanced OCR techniques and machine learning algorithms to further improve text recognition accuracy.
- Investigate methods for handling complex document layouts, fonts, and languages to enhance the robustness of the system.
- Develop user-friendly interfaces and applications for automating document analysis and recognition tasks in real-world scenarios.

## Usage

1. Clone this repository.
2. Install the required dependencies.
3. Run the provided scripts for each task to replicate the analysis and results.
4. Explore the generated PDFs to observe the differences in text recognition between skewed and deskewed documents.
