# Intelligent Data Room

A multi-agent AI system that allows you to upload CSV/Excel files and have natural language conversations with your data.

## 🌟 Features

- **Multi-Agent System**: Separate "Thinking" (Planner) and "Doing" (Executor) agents
- **Natural Language Queries**: Ask questions about your data in plain English
- **Automatic Visualizations**: Charts are generated based on your questions
- **Context Retention**: Remembers last 5 conversations for follow-up questions
- **File Support**: CSV and Excel files up to 10MB

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────┐
│   Agent 1: Planner          │
│   - Analyzes question       │
│   - Reviews data schema     │
│   - Creates execution plan  │
│   - Determines chart type   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Agent 2: Executor         │
│   - Executes the plan       │
│   - Uses PandasAI + Gemini  │
│   - Generates response      │
│   - Creates visualizations  │
└─────────────┬───────────────┘
              │
              ▼
         Response + Chart
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Gemini API key

### Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd csv_chatbot
```

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Set up your environment:

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

1. Run the application:

```bash
streamlit run app.py
```

## 📁 Project Structure

```
csv_chatbot/
├── app.py                 # Main Streamlit application
├── agents/
│   ├── planner.py        # Agent 1: Planning and analysis
│   └── executor.py       # Agent 2: Execution and response
├── utils/
│   ├── data_loader.py    # File validation and loading
│   ├── memory.py         # Conversation context retention
│   └── visualizations.py # Chart generation with Plotly
├── requirements.txt
├── .env.example
└── README.md
```

## 💬 Sample Prompts

### Easy

- "Create a bar chart showing total Sales by Category"
- "Show the distribution of Sales across Regions with a pie chart"
- "Which Customer Segment places the most orders?"
- "Top 5 States by total Sales"

### Medium

- "Which Sub-Categories are unprofitable on average?"
- "Compare Sales trends of different Ship Modes over time"
- "Is there a correlation between Discount and Profit?"
- "Show the Top 10 Customers by Profit"

### Follow-up

- "Who are the top 5 customers?" → "Show their locations"

## 🔧 Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your Google Gemini API key |

### Memory Settings

The system retains the last 5 conversation turns by default. This can be configured in `utils/memory.py`.

## 🎨 Tech Stack

- **Frontend**: Streamlit
- **AI/LLM**: Google Gemini API, PandasAI
- **Data**: Pandas, OpenPyXL
- **Visualization**: Plotly

## 📝 License

MIT License
