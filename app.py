import warnings
import streamlit as st
import pandas as pd
import logging
from dotenv import load_dotenv

# Import custom modules
from utils.data_loader import (
    validate_file,
    load_data,
    get_schema_info,
    format_schema_for_prompt,
)
from utils.memory import ConversationMemory
from utils.visualizations import create_chart
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CommAI",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
        /* Import font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        /* Force font globally */
        html, body, [class*="css"], .stMarkdown, h1, h2, h3, p, div, label, input, textarea {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        }
        
        /* Chat message styling */
        .user-message {
            background: linear-gradient(135deg, #96e6a1 0%, #d4fc79 100%);
            padding: 1rem 1.5rem;
            border-radius: 1rem 1rem 0.25rem 1rem;
            margin: 0.5rem 0;
            color: #0F172A;
        }
        
        .assistant-message {
            background: #1E293B;
            padding: 1rem 1.5rem;
            border-radius: 1rem 1rem 1rem 0.25rem;
            margin: 0.5rem 0;
            border: 1px solid #334155;
        }
        
        /* Plan expander styling */
        .plan-container {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 0.75rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* File uploader styling */
        [data-testid="stFileUploader"] {
            background: #1E293B;
            border-radius: 0.75rem;
            padding: 1rem;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #96e6a1 0%, #d4fc79 100%);
            color: #0F172A;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
        }
        
        /* Title styling */
        .main-title {
            font-size: 5rem !important;
            font-weight: 700;
            background: linear-gradient(135deg, #96e6a1 0%, #d4fc79 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            text-align: center;
            color: #94A3B8;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        /* Schema info card */
        .schema-card {
            background: #1E293B;
            border-radius: 0.75rem;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #334155;
        }
        
        /* Agent badge styling */
        .agent-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }
        
        .agent-planner {
            background: linear-gradient(135deg, #96e6a1 0%, #d4fc79 100%);
            color: #0F172A;
        }
        
        .agent-executor {
            background: linear-gradient(135deg, #22C55E 0%, #14B8A6 100%);
            color: white;
        }

        /* Target the specific delete button inside the uploader */
        [data-testid="stFileUploaderDeleteBtn"] > button {
            width: 28px !important;
            height: 28px !important;
            min-width: 28px !important;
            min-height: 28px !important;
            max-width: 28px !important;
            max-height: 28px !important;
            padding: 0px !important;
            line-height: 1 !important;
            margin: 0px !important;
            border-radius: 8px !important;
            border: none !important;
            background-color: transparent !important;
            transition: all 0.2s ease-in-out;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        [data-testid="stFileUploaderDeleteBtn"] > button:hover {
            background-color: #333c4e !important;
        }

        /* Ensure the 'X' icon itself remains visible (Red) */
        [data-testid="stFileUploaderDeleteBtn"] > button svg {
            fill: #9c9d9f !important;
        }

        [data-testid="stChatInput"] > div:focus-within {
            border: 0.67px solid #96e6a1 !important;
        }

        .stAlert {
            background-color: rgba(30, 41, 59, 0.7) !important;
            color: #96e6a1 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_turns=5)
    if "df" not in st.session_state:
        st.session_state.df = None
    if "schema" not in st.session_state:
        st.session_state.schema = None
    if "planner" not in st.session_state:
        st.session_state.planner = None
    if "executor" not in st.session_state:
        st.session_state.executor = None
    if "agents_initialized" not in st.session_state:
        st.session_state.agents_initialized = False
    if "dynamic_prompts" not in st.session_state:
        st.session_state.dynamic_prompts = []
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = None


def initialize_agents():
    """Initialize the AI agents."""
    # Check if agents are initialized and have required methods
    planner_ready = st.session_state.agents_initialized and hasattr(
        st.session_state.planner, "generate_sample_prompts"
    )

    if not planner_ready:
        try:
            st.session_state.planner = PlannerAgent()
            st.session_state.executor = ExecutorAgent()
            st.session_state.agents_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing agents: {e}")
            return False
    return True


def render_sidebar():
    """Render the sidebar with file upload and data preview."""
    selected_sample = None

    with st.sidebar:
        st.markdown("## Data Upload")

        uploaded_file = st.file_uploader(
            "Upload your CSV or Excel file",
            type=["csv", "xlsx", "xls"],
        )

        if uploaded_file:
            # Validate file
            is_valid, error_msg = validate_file(uploaded_file)

            if not is_valid:
                st.error(error_msg)
            else:
                # Load data
                df, load_error = load_data(uploaded_file)

                if load_error:
                    st.error(load_error)
                else:
                    if st.session_state.last_file_name != uploaded_file.name:
                        st.session_state.df = df
                        st.session_state.schema = get_schema_info(df)
                        st.session_state.memory.set_file(uploaded_file.name)
                        st.session_state.last_file_name = uploaded_file.name

                        # Generate dynamic prompts
                        with st.spinner("Analyzing data to suggest questions..."):
                            schema_str = format_schema_for_prompt(
                                st.session_state.schema
                            )
                            st.session_state.dynamic_prompts = (
                                st.session_state.planner.generate_sample_prompts(
                                    schema_str
                                )
                            )

                    # Show schema info
                    st.markdown("### Schema")
                    schema = st.session_state.schema
                    st.markdown(f"**Rows:** {schema['row_count']:,}")
                    st.markdown(f"**Columns:** {schema['column_count']}")

                    with st.expander("View Column Details"):
                        for col in schema["columns"]:
                            st.markdown(f"- **{col['name']}** ({col['dtype']})")

        # Sample prompts
        if st.session_state.df is not None and st.session_state.dynamic_prompts:
            st.markdown("### Try These Prompts")
            for prompt in st.session_state.dynamic_prompts:
                if st.button(prompt, key=f"sample_{prompt}", width="stretch"):
                    selected_sample = prompt
            if selected_sample:
                return selected_sample
        elif st.session_state.df is None:
            st.success("Upload a file to see suggested questions.")

    return None


def process_question(question: str):
    """Process a user question through the multi-agent system."""
    logger.info(f"Processing question: {question}")
    if st.session_state.df is None:
        return {
            "response": "Please upload a data file first.",
            "plan": None,
            "chart": None,
        }

    # Get schema info and context
    schema_str = format_schema_for_prompt(st.session_state.schema)
    context = st.session_state.memory.get_context()

    result = {"response": "", "plan": None, "chart": None, "plan_text": ""}

    # Step 1: Planner Agent creates the plan
    with st.spinner("Planner Agent: Analyzing your question..."):
        plan = st.session_state.planner.create_plan(
            question=question, schema_info=schema_str, conversation_context=context
        )
        result["plan"] = plan
        result["plan_text"] = st.session_state.planner.get_reasoning(plan)

    # Step 2: Executor Agent executes the plan
    with st.spinner("Executor Agent: Processing data..."):
        execution_result = st.session_state.executor.execute(
            df=st.session_state.df, plan=plan, original_question=question
        )
        result["response"] = execution_result.get("response", "Analysis complete.")

    # Step 3: Generate chart if needed
    viz_type = plan.get("visualization", "NONE").upper()
    if viz_type != "NONE":
        with st.spinner("Generating visualization..."):
            chart_config = plan.get("chart_config", {})
            chart_data = execution_result.get("chart_data")

            # For scatter plots (correlation), use raw data if no chart_data
            if viz_type == "SCATTER":
                x_col = chart_config.get("x_column")
                y_col = chart_config.get("y_column")

                # Use the original dataframe for scatter plots
                scatter_df = st.session_state.df

                # Try to find columns by name
                scatter_cols = list(scatter_df.columns)
                if not x_col or x_col not in scatter_cols:
                    for col in scatter_cols:
                        if "discount" in col.lower():
                            x_col = col
                            break
                if not y_col or y_col not in scatter_cols:
                    for col in scatter_cols:
                        if "profit" in col.lower():
                            y_col = col
                            break

                if x_col and y_col and x_col in scatter_cols and y_col in scatter_cols:
                    fig = create_chart(
                        df=scatter_df,
                        chart_type="scatter",
                        x=x_col,
                        y=y_col,
                        title=chart_config.get("title", f"{x_col} vs {y_col}"),
                    )
                    result["chart"] = fig

            elif (
                chart_data is not None
                and isinstance(chart_data, pd.DataFrame)
                and len(chart_data) > 0
            ):
                # Auto-detect columns if not specified or not found
                x_col = chart_config.get("x_column")
                y_col = chart_config.get("y_column")

                # Get actual column names from the result data
                result_columns = list(chart_data.columns)

                # If specified columns don't exist in result, auto-detect
                if not x_col or x_col not in result_columns:
                    # First non-numeric column is usually the category/x-axis
                    for col in result_columns:
                        if (
                            chart_data[col].dtype == "object"
                            or "name" in col.lower()
                            or "date" in col.lower()
                            or "category" in col.lower()
                            or "year" in col.lower()
                            or "month" in col.lower()
                        ):
                            x_col = col
                            break
                    if not x_col:
                        x_col = result_columns[0]

                if not y_col or y_col not in result_columns:
                    # First numeric column is usually the value/y-axis
                    for col in result_columns:
                        if col != x_col and chart_data[col].dtype in [
                            "int64",
                            "float64",
                        ]:
                            y_col = col
                            break
                    if not y_col and len(result_columns) > 1:
                        y_col = result_columns[1]

                # Create the chart
                fig = create_chart(
                    df=chart_data,
                    chart_type=plan["visualization"],
                    x=x_col,
                    y=y_col,
                    title=chart_config.get("title"),
                    color=chart_config.get("color_by")
                    if chart_config.get("color_by") in result_columns
                    else None,
                )
                result["chart"] = fig

    # Add to memory
    st.session_state.memory.add_turn(
        question=question,
        plan=result["plan_text"],
        response=result["response"],
        chart_type=plan.get("visualization"),
    )

    return result


def render_message(message: dict, key: str):
    """Render a single chat message."""
    if message["role"] == "user":
        st.markdown(
            f"""
        <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
            <div class="user-message">
                {message["content"]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        # Assistant message with response
        response_text = message.get("response", message.get("content", ""))

        # Use Streamlit's native components for proper markdown rendering
        with st.container():
            st.markdown(response_text)

        # Show execution plan
        if message.get("plan_text"):
            with st.expander("View Execution Plan", expanded=False):
                st.markdown(message["plan_text"])

        # Show chart
        if message.get("chart"):
            st.plotly_chart(message["chart"], width="stretch", key=f"chart_{key}")


def main():
    """Main application entry point."""
    init_session_state()

    # Initialize agents
    if not initialize_agents():
        st.error("Failed to initialize AI agents. Please check your API key.")
        return

    # Header
    st.markdown('<h1 class="main-title">CommAI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload your data and start a conversation with AI-powered analysis</p>',
        unsafe_allow_html=True,
    )

    # Returns sample prompt if clicked
    sample_prompt = render_sidebar()

    # Main chat area
    if st.session_state.df is not None:
        # Chat interface
        chat_container = st.container()

        # Display existing messages
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                render_message(msg, f"msg_{i}")

        # Handle sample prompt click
        if sample_prompt:
            st.session_state.messages.append({"role": "user", "content": sample_prompt})

            result = process_question(sample_prompt)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "response": result["response"],
                    "plan_text": result["plan_text"],
                    "chart": result["chart"],
                }
            )

            st.rerun()

        # Chat input
        user_input = st.chat_input("Ask a question about your data...")

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Process question
            result = process_question(user_input)

            # Add assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "response": result["response"],
                    "plan_text": result["plan_text"],
                    "chart": result["chart"],
                }
            )

            st.rerun()


if __name__ == "__main__":
    main()
