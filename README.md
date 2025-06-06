# Competitor Basic Financial Analysis AI-Agent

#### Demo Video
https://github.com/user-attachments/assets/f7a80539-cb79-4792-9393-7172fbd6c2ee

## Overview

The **Competitor Basic Financial Analysis AI-Agent** is a Streamlit-based web application designed to compare the financial performance of MyAICo.AI with its competitors (e.g., Google, Microsoft, Meta). The app leverages AI to gather, analyze, and compare financial data, providing a comprehensive report with insights and recommendations. It uses LangGraph to orchestrate a multi-step workflow, yfinance to fetch competitor financial data, and Hugging Face's Mistral-7B model for financial analysis and report generation.

## Features

- **Upload Financial Data**: Upload a CSV file containing MyAICo.AI's financial data.
- **Competitor Analysis**: Fetch financial data for competitors using yfinance.
- **AI-Powered Insights**: Analyze financial data and generate comparisons using Hugging Face's Mistral-7B model.
- **Multi-Step Workflow**: Uses LangGraph to orchestrate a pipeline of tasks (gather financials, analyze data, research competitors, compare performance, collect feedback, and write a report).
- **Downloadable Reports**: Generate and download a comprehensive financial report in Markdown format.
- **Customizable**: Specify the task, competitors, and maximum revisions for iterative feedback.

## Tech Stack

- **Frontend**: Streamlit (`streamlit==1.43.2`) - For the web interface.
- **Backend**:
  - Python 3.8+ - Core programming language.
  - LangGraph (`langgraph==0.3.18`) - For orchestrating the multi-step workflow.
  - yfinance (`yfinance==0.2.55`) - To fetch financial data for competitors.
  - Hugging Face API (`requests==2.32.3`) - For financial analysis using the Mistral-7B-Instruct-v0.2 model.
  - pandas (`pandas==2.2.3`) - For processing CSV data.
  - python-dotenv (`python-dotenv==1.0.1`) - For managing environment variables.
  - typing-extensions (`typing-extensions==4.12.2`) - For type hints in LangGraph.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- A Hugging Face API key

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abdllhh/competitor_basic_financial_analysis_AI-agent.git
   cd competitor_basic_financial_analysis_AI-agent

2. **Setup a Vitrual Env**
   ```
   python -m venv venv
   ```
   ### On Windows:
   ```
   venv\Scripts\activate
   ```
   ### On macOS/Linux:
   ```
   source venv/bin/activate
   
4. **Install Dependencies**
   ```
   pip install -r requirements.txt

5. **Configure env variables**
   create  .env file
   ```bash
   echo HUGGINGFACE_API_KEY=your-hugging-face-api-key > .env

6. **Run the app**
   ```
   streamlit run financial_agent.py

