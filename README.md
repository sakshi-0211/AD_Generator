# Creative Ad Copy Generator

## Overview
The **Creative Ad Copy Generator** is a powerful tool designed to help marketers and businesses create compelling ad copies and visuals effortlessly. Built with **Streamlit**, this application leverages **Gemini AI** for generating ad text and **Stability AI** for creating visually stunning ad images. Whether you're promoting a product, service, or campaign, this tool provides a seamless experience for generating professional-quality ads tailored to your target audience and tone.

## Features

### Ad Copy Generation
- Generate ad headlines, main copy, and call-to-action (CTA) using **Gemini AI**.
- Tailored for specific audiences and tones (e.g., **Professional, Friendly, Humorous, Urgent, Inspirational, Luxurious**).

### Ad Image Generation
- Create visually appealing ad images using **Stability AI**.
- Tone-specific styling, including fonts, colors, and decorative elements.

### User-Friendly Interface
- Input product details, target audience, tone, and ad dimensions.
- View generated ad copy and visuals in a clean and intuitive interface.

### Download Options
- Download generated ad images in **PNG** format.
- Download ad copy in **JSON** format for easy integration into marketing campaigns.

### Dataset Integration
- Retrieve similar ad examples from a dataset to guide AI-generated content.
- Fallback dataset ensures functionality even if the primary dataset fails to load.

## Technologies Used

### Frontend
- **Streamlit**: For building the interactive web interface.

### Backend
- **Python**: Core programming language.
- **Gemini AI**: For generating ad copy using advanced natural language processing.
- **Stability AI**: For generating high-quality ad visuals.

### Libraries
- **Pandas**: For handling and retrieving ad examples from the dataset.
- **Pillow (PIL)**: For image processing and applying tone-specific styling.
- **Base64**: For encoding and decoding images for download functionality.
- **JSON**: For formatting and parsing ad copy data.

## Installation and Setup

### Prerequisites
- **Python 3.8 or higher**
- **Streamlit**
- **Gemini AI API key**
- **Stability AI API key**

### Steps

#### Clone the Repository
```bash
git clone https://github.com/your-username/creative-ad-copy-generator.git
cd creative-ad-copy-generator
```


#### Set Up API Keys
Obtain your **Gemini AI** and **Stability AI** API keys.
Create a `.env` file in the root directory and add your keys:
```plaintext
GEMINI_API_KEY=your_gemini_api_key
STABILITY_API_KEY=your_stability_api_key
```

#### Run the Application
```bash
streamlit run app.py
```

## Usage

### Input Product Details
- Enter the product name, description, target audience, and ad dimensions.
- Select the desired tone for the ad (**e.g., Professional, Friendly, Humorous**).

### Generate Ad
- Click the **"Generate Ad"** button to create ad copy and visuals.

### View and Download
- View the generated ad copy and image in the interface.
- Download the **ad image (PNG)** and **ad copy (JSON)** using the provided buttons.

## Future Enhancements
- **RAG (Retrieval-Augmented Generation)**: Improve ad copy generation by retrieving relevant examples from the dataset and using them as context.
- **Advanced Tone Customization**: Add more nuanced tone-specific styling options, such as font weight, spacing, and advanced decorative elements.
- **Improved Error Handling**: Enhance error handling for API failures and edge cases to ensure a seamless user experience.
- **Performance Optimization**: Streamline the codebase for faster processing and better scalability.

## Contributing
We welcome contributions to this project! If you'd like to contribute, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bugfix.
3. **Commit your changes** and push to the branch.
4. **Submit a pull request** with a detailed description of your changes.


## Acknowledgments
- **Gemini AI** for providing the natural language processing capabilities.
- **Stability AI** for enabling high-quality image generation.
- **Streamlit** for the intuitive web interface framework.

---

### 🚀 Happy Coding!
