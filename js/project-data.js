const projectsData = {
    "churn": {
        title: "Customer Churn Prediction",
        tagline: "Leveraging Advanced Machine Learning to Retain Valued Customers",
        year: "2023",
        category: "Machine Learning",
        tech: "Python",
        description: `
            <p>In the highly competitive telecommunications sector, retaining customers is crucial. This project addresses this challenge by developing a robust predictive model that identifies customers at high risk of churning. By analyzing historical usage patterns, demographic data, and service interaction logs, the model provides actionable insights that allow the business to intervene proactively.</p>
            <p>The solution was deployed as a real-time API service, enabling the customer success team to receive instant alerts and tailored retention offers for at-risk users, resulting in a measurable decrease in monthly churn rates.</p>
        `,
        features: [
            {
                title: "High Accuracy Ensemble",
                desc: "Utilized an ensemble of XGBoost and Random Forest models to achieve a 92% ROC-AUC score, significantly outperforming baseline logistic regression."
            },
            {
                title: "Explainable AI",
                desc: "Integrated SHAP (SHapley Additive exPlanations) values to provide transparency, showing exactly <i>why</i> a specific customer was flagged as high-risk."
            },
            {
                title: "Real-time Dashboard",
                desc: "Built an interactive dashboard for stakeholders to monitor churn trends, filter by user segments, and track the effectiveness of retention campaigns."
            },
            {
                title: "Automated Pipeline",
                desc: "Implemented a fully automated CI/CD pipeline for model retraining and deployment using AWS SageMaker and GitHub Actions."
            }
        ],
        techStack: ["Python", "Scikit-learn", "XGBoost", "Pandas", "Flask API", "AWS SageMaker", "Docker", "Streamlit"],
        links: {
            github: "#",
            demo: "#"
        }
    },
    "vision": {
        title: "Image Classification System",
        tagline: "Deep Learning Powered Visual Recognition for Automated Sorting",
        year: "2023",
        category: "Computer Vision",
        tech: "PyTorch",
        description: `
            <p>This project implements a state-of-the-art image classification system designed for industrial quality control automation. Leveraging deep convolutional neural networks (CNNs), the system automatically categorizes product images into defect-free and defective classes with high precision.</p>
            <p>The model was trained using transfer learning techniques on the ResNet50 architecture, allowing it to achieve high accuracy even with a limited labeled dataset. The final system is containerized and deployed on edge devices for real-time inference on the factory floor.</p>
        `,
        features: [
            {
                title: "Transfer Learning",
                desc: "Fine-tuned a pre-trained ResNet50 model to adapt to specific domain data, reducing training time and computational resources."
            },
            {
                title: "Data Augmentation",
                desc: "Implemented advanced augmentation techniques (rotation, flip, scaling) to improve model generalization and robustness against varying lighting conditions."
            },
            {
                title: "Edge Deployment",
                desc: "Optimized the model using TensorRT for low-latency inference on NVIDIA Jetson edge devices."
            }
        ],
        techStack: ["PyTorch", "OpenCV", "ResNet50", "Docker", "NVIDIA Jetson", "Python"],
        links: {
            github: "#",
            demo: "#"
        }
    },
    "nlp": {
        title: "NLP Sentiment Analyzer",
        tagline: "Understanding Social Media Sentiment with Transformers",
        year: "2022",
        category: "NLP",
        tech: "Transformers",
        description: `
            <p>In the age of social media, understanding public opinion is vital for brand management. This tool analyzes social media feeds to determine the sentiment (positive, negative, neutral) regarding specific topics or brand mentions in real-time.</p>
            <p>Built upon the BERT architecture, the model understands context and nuance better than traditional bag-of-words approaches. It processes thousands of tweets per minute and visualizes the aggregate sentiment trends over time.</p>
        `,
        features: [
            {
                title: "Transformer Architecture",
                desc: "Uses a fine-tuned BERT model for state-of-the-art accuracy in understanding context and sarcasm."
            },
            {
                title: "Real-time Processing",
                desc: "Ingests and processes live Twitter/X data streams using Apache Kafka and Spark Streaming."
            },
            {
                title: "Interactive Visualization",
                desc: "frontend dashboard built with React and D3.js to visualize sentiment shifts and trending keywords."
            }
        ],
        techStack: ["Python", "BERT", "Hugging Face", "FastAPI", "React", "Kafka"],
        links: {
            github: "#",
            demo: "#"
        }
    },
    "sales": {
        title: "Sales Forecasting Dashboard",
        tagline: "Predicting Future Revenue with Time Series Analysis",
        year: "2022",
        category: "Data Analytics",
        tech: "Time Series",
        description: `
            <p>Accurate sales forecasting is key to inventory management and resource planning. This project provides a comprehensive dashboard that predicts future sales based on historical data, seasonality, and market trends.</p>
            <p>Using Long Short-Term Memory (LSTM) recurrent neural networks, the model captures complex temporal dependencies. The results are presented in an intuitive dashboard that allows business managers to run 'what-if' scenarios.</p>
        `,
        features: [
            {
                title: "LSTM Networks",
                desc: "Implemented Recurrent Neural Networks (RNN) specifically designed to learn long-term dependencies in time-series data."
            },
            {
                title: "Interactive Scenarios",
                desc: "Users can adjust parameters (e.g., marketing spend, pricing) to see potential impacts on future sales."
            },
            {
                title: "Automated Reporting",
                desc: "Generates weekly PDF reports summarizing forecast accuracy and highlighting significant deviations."
            }
        ],
        techStack: ["Python", "Keras/TensorFlow", "Pandas", "Streamlit", "PostgreSQL", "Plotly"],
        links: {
            github: "#",
            demo: "#"
        }
    }
};
