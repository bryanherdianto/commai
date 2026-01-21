# CommAI - CSV Chatbot

![CommAI GitHub Banner](https://i.imgur.com/La8Hbmz.png)

<p align="center">
 <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
 <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
 <img src="https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" />
 <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
 <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
</p>

CommAI is a multi-agent system designed for conversational data analysis. It allows users to upload CSV or Excel files and interact with their data using natural language.

## Features

- Multi-Agent Architecture: Utilizes a dedicated Planner Agent for strategy and an Executor Agent for data processing.
- Dynamic Prompt Suggestions: Automatically generates relevant starting questions based on the uploaded data schema.
- Automated Visualizations: Supports various chart types including bar, line, pie, scatter, and maps based on the query context.
- Context Awareness: Maintains conversation history to support follow-up questions and multi-turn analysis.
- File Support: Validates and processes CSV and Excel files with a customizable 10 MB size limit.
- Logging and Error Handling: Integrated logging for tracking agent performance and system diagnostics.

## System Design

### Logic Flowchart

![Flowchart](https://i.imgur.com/e88MbiS.png)

### Agent Workflow

![Agent Workflow Diagram](https://i.imgur.com/YPhXJJR.png)

### System Architecture

![Architecture Diagram](https://i.imgur.com/q5xe19W.png)

## System Workflow

The application follows a structured orchestration pattern:

1. Data Ingestion: The user uploads a file which is validated and loaded into a pandas DataFrame.
2. Initial Analysis: The Planner Agent analyzes the schema and suggests relevant insight-driven prompts.
3. Planning: Upon receiving a query, the Planner Agent creates a step-by-step execution strategy.
4. Execution: The Executor Agent generates and runs Python code to process the data and prepare visualization subsets.
5. Presentation: The system renders a text-based response alongside interactive Plotly charts.

## Project Structure

```txt
csv_chatbot/
├── app.py                # Main Streamlit application and UI orchestration
├── agents/
│   ├── planner.py        # Logic for query analysis and plan generation
│   └── executor.py       # Logic for code generation and data execution
├── utils/
│   ├── data_loader.py    # File validation, encoding detection, and loading
│   ├── geocoding.py      # Location-to-coordinate conversion for map charts
│   ├── memory.py         # Conversation state and context management
│   └── visualizations.py # Plotly chart generation and branding
├── .streamlit/
│   └── config.toml       # Streamlit server configuration (size limits)
├── requirements.txt      # List of project dependencies
└── README.md             # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.11.14
- Google Gemini API Key

### Installation

1. Clone the repository:

     ```bash
     git clone https://github.com/yourusername/csv_chatbot.git
     cd csv_chatbot
     ```

2. Create and activate a virtual environment:

     ```bash
     python -m venv venv
     # On Windows:
     venv\Scripts\activate
     # On Linux/macOS:
     source venv/bin/activate
     ```

3. Install the required packages:

     ```bash
     pip install -r requirements.txt
     ```

4. Configure environment variables:
Create a .env file in the root directory and add your API key:

     ```env
     GEMINI_API_KEY=your_google_gemini_api_key
     ```

5. Run the application:

     ```bash
     streamlit run app.py
     ```

## Tech Stack

- Framework: Streamlit
- Language Models: Google Gemini API
- Data Analysis: Pandas
- Mapping: Geopy
- Visualizations: Plotly

## Live Website

Access the live application here: [https://commai.streamlit.app/](https://commai.streamlit.app/)

## Video Demo

A comprehensive video demonstration explaining the multi-agent architecture and system functionality is available here: [https://youtu.be/7NdGish7pVk](https://youtu.be/7NdGish7pVk)
