import os
import streamlit as st
import pandas as pd
import logging
from google import genai
from google.genai import types
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)


class ExecutorAgent:
    """
    Agent 2: The Executor
    Executes plans using Gemini-powered code generation and PandasAI concepts.
    """

    SYSTEM_PROMPT = """
    You are a data analysis expert using PandasAI concepts. Given a pandas DataFrame and a question,
    write Python code to answer the question. 

    IMPORTANT RULES:
    1. The DataFrame is already loaded as 'df'
    2. Write ONLY executable Python code, no markdown, no explanations, no comments
    3. Store your final result in a variable called 'result'
    4. The result should be either:
    - A pandas DataFrame for tabular results
    - A single value (number, string) for simple answers
    5. Use proper pandas operations
    6. Always include .reset_index() when using groupby to get a clean DataFrame
    7. For top N results, use .head(N) after sorting

    Example for "Top 5 customers by Sales":
    result = df.groupby('Customer Name')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(5)
    """

    def __init__(self):
        """Initialize the Executor agent with Gemini API."""
        self.api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Initialize google.genai client
        self.client = genai.Client(api_key=self.api_key)
        self.model_id = "gemini-flash-lite-latest"

    def execute(
        self, df: pd.DataFrame, plan: Dict[str, Any], original_question: str
    ) -> Dict[str, Any]:
        """
        Execute the plan on the DataFrame using Gemini-powered code generation.

        Args:
            df: pandas DataFrame with the data
            plan: Execution plan from Planner agent
            original_question: The user's original question

        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing: {original_question}")
        result = {
            "success": False,
            "response": "",
            "data": None,
            "chart_data": None,
            "visualization": plan.get("visualization", "NONE"),
            "chart_config": plan.get("chart_config", {}),
            "error": None,
        }

        try:
            # Generate code using Gemini
            code = self._generate_code(df, plan, original_question)

            if code:
                logger.info("Generated execution code")
                # Execute the generated code
                exec_result = self._execute_code(code, df)

                if exec_result["success"]:
                    logger.info("Code execution successful")
                    data = exec_result["result"]

                    if isinstance(data, pd.DataFrame):
                        result["data"] = data
                        result["response"] = self._format_dataframe_response(data, plan)
                        result["chart_data"] = data
                    elif isinstance(data, (int, float)):
                        result["response"] = self._format_numeric_response(data, plan)
                        result["data"] = data
                    elif isinstance(data, pd.Series):
                        df_result = data.reset_index()
                        df_result.columns = [df_result.columns[0], "Value"]
                        result["data"] = df_result
                        result["response"] = self._format_dataframe_response(
                            df_result, plan
                        )
                        result["chart_data"] = df_result
                    else:
                        result["response"] = str(data)
                        result["data"] = data

                    result["success"] = True
                else:
                    logger.warning(
                        f"Initial execution failed: {exec_result.get('error')}"
                    )
                    # Try a simpler approach if first attempt fails
                    simple_code = self._generate_simple_code(
                        df, plan, original_question
                    )
                    if simple_code:
                        exec_result = self._execute_code(simple_code, df)
                        if exec_result["success"]:
                            data = exec_result["result"]
                            if isinstance(data, pd.DataFrame):
                                result["data"] = data
                                result["response"] = self._format_dataframe_response(
                                    data, plan
                                )
                                result["chart_data"] = data
                                result["success"] = True
                            elif isinstance(data, pd.Series):
                                df_result = data.reset_index()
                                result["data"] = df_result
                                result["response"] = self._format_dataframe_response(
                                    df_result, plan
                                )
                                result["chart_data"] = df_result
                                result["success"] = True

                        if not result["success"]:
                            logger.error(
                                f"Execution failed after fallback: {exec_result.get('error')}"
                            )
                            result["error"] = exec_result["error"]
                            result["response"] = (
                                f"Error executing analysis: {exec_result['error']}"
                            )
            else:
                logger.error("Generated code was empty")
                result["response"] = "Could not generate analysis code."

        except Exception as e:
            logger.error(
                f"Unexpected error in ExecutorAgent.execute: {e}", exc_info=True
            )
            result["error"] = str(e)
            result["response"] = f"Error during analysis: {str(e)}"

        return result

    def _generate_code(
        self, df: pd.DataFrame, plan: Dict[str, Any], question: str
    ) -> Optional[str]:
        """Generate Python code to answer the question using Gemini."""

        # Get column info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            columns_info.append(f"- {col} ({dtype}): {sample}")

        columns_str = "\n".join(columns_info)

        # Build steps text
        steps = plan.get("steps", [])
        steps_text = (
            "\n".join([f"- {step}" for step in steps]) if steps else "Analyze as needed"
        )

        prompt = f"""
        DataFrame columns and sample values:
        {columns_str}

        Question: {question}

        Analysis steps to follow:
        {steps_text}

        Write Python pandas code to answer this question. Store the final result in 'result'.
        The DataFrame is already loaded as 'df'. Write ONLY executable code.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )

            code = response.text

            # Clean up code
            code = code.replace("```python", "").replace("```", "").strip()

            # Remove any lines that start with #
            lines = [
                line for line in code.split("\n") if not line.strip().startswith("#")
            ]
            code = "\n".join(lines)

            return code

        except Exception as e:
            print(f"Code generation error: {e}")
            return None

    def _generate_simple_code(
        self, df: pd.DataFrame, plan: Dict[str, Any], question: str
    ) -> Optional[str]:
        """Generate simpler code as fallback."""

        # Try to extract column names from the question
        question_lower = question.lower()

        # Common patterns for Top N customers
        if "top" in question_lower and "customer" in question_lower:
            if "profit" in question_lower:
                return "result = df.groupby('Customer Name')['Profit'].sum().reset_index().sort_values('Profit', ascending=False).head(5)"
            elif "sales" in question_lower:
                return "result = df.groupby('Customer Name')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(5)"

        # Category-based analysis
        if "category" in question_lower and "sub" not in question_lower:
            if "profit" in question_lower and "sales" in question_lower:
                return "result = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()"
            elif "profit" in question_lower:
                return "result = df.groupby('Category')['Profit'].sum().reset_index()"
            elif "sales" in question_lower:
                return "result = df.groupby('Category')['Sales'].sum().reset_index()"

        # Sub-Category analysis
        if "sub-categor" in question_lower or "subcategor" in question_lower:
            if (
                "unprofitable" in question_lower
                or "negative" in question_lower
                or "loss" in question_lower
            ):
                return """
                result = df.groupby('Sub-Category')['Profit'].mean().reset_index()
                result = result[result['Profit'] < 0].sort_values('Profit')
                result.columns = ['Sub-Category', 'Average Profit']
                """
            elif "profit" in question_lower:
                return "result = df.groupby('Sub-Category')['Profit'].sum().reset_index().sort_values('Profit')"
            elif "sales" in question_lower:
                return "result = df.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)"

        # Region-based analysis including return rate
        if "region" in question_lower:
            if "return" in question_lower or "rate" in question_lower:
                return """
                if 'Returned' in df.columns:
                    region_orders = df.groupby('Region')['Order ID'].nunique()
                    region_returns = df[df['Returned'] == 'Yes'].groupby('Region')['Order ID'].nunique()
                    result = pd.DataFrame({
                        'Region': region_orders.index,
                        'Total Orders': region_orders.values,
                        'Returns': region_returns.reindex(region_orders.index, fill_value=0).values
                    })
                    result['Return Rate (%)'] = (result['Returns'] / result['Total Orders'] * 100).round(2)
                else:
                    result = df.groupby('Region')['Order ID'].nunique().reset_index()
                    result.columns = ['Region', 'Order Count']
                    result['Note'] = 'Return data not available'
                """
            elif "sales" in question_lower:
                return "result = df.groupby('Region')['Sales'].sum().reset_index()"
            elif "profit" in question_lower:
                return "result = df.groupby('Region')['Profit'].sum().reset_index()"

        # Year-based analysis (2018-2021 style)
        if (
            "year" in question_lower
            or "2018" in question_lower
            or "2019" in question_lower
            or "2020" in question_lower
            or "2021" in question_lower
        ):
            if "profit" in question_lower:
                return """
                df['Order Date'] = pd.to_datetime(df['Order Date'])
                df['Year'] = df['Order Date'].dt.year
                result = df.groupby('Year')['Profit'].sum().reset_index()
                result = result.sort_values('Year')
                """
            elif "sales" in question_lower:
                return """
                df['Order Date'] = pd.to_datetime(df['Order Date'])
                df['Year'] = df['Order Date'].dt.year
                result = df.groupby('Year')['Sales'].sum().reset_index()
                result = result.sort_values('Year')
                """

        # Time series / trend analysis (monthly)
        if (
            "trend" in question_lower
            or "over time" in question_lower
            or "by month" in question_lower
        ):
            if "sales" in question_lower:
                return """
                df['Order Date'] = pd.to_datetime(df['Order Date'])
                df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
                result = df.groupby('YearMonth')['Sales'].sum().reset_index()
                result = result.sort_values('YearMonth')
                """
            elif "profit" in question_lower:
                return """
                df['Order Date'] = pd.to_datetime(df['Order Date'])
                df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
                result = df.groupby('YearMonth')['Profit'].sum().reset_index()
                result = result.sort_values('YearMonth')
                """

        # Correlation analysis
        if "correlation" in question_lower or (
            "discount" in question_lower and "profit" in question_lower
        ):
            return """
            correlation = df['Discount'].corr(df['Profit'])
            result = round(correlation, 2)
            """

        # Segment analysis
        if "segment" in question_lower:
            if "order" in question_lower or "count" in question_lower:
                return "result = df.groupby('Segment')['Order ID'].count().reset_index().rename(columns={'Order ID': 'Order Count'})"
            elif "sales" in question_lower:
                return "result = df.groupby('Segment')['Sales'].sum().reset_index()"

        # State analysis
        if "state" in question_lower:
            if "top" in question_lower or "5" in question_lower:
                return "result = df.groupby('State')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(5)"

        return None

    def _execute_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Safely execute the generated code."""
        try:
            # Create execution environment with pandas
            import numpy as np

            local_vars = {"df": df, "pd": pd, "np": np}

            # Execute the code
            exec(code, {"pd": pd, "np": np, "__builtins__": __builtins__}, local_vars)

            # Get the result
            result = local_vars.get("result", None)

            return {"success": True, "result": result, "error": None}

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}

    def _format_dataframe_response(self, df: pd.DataFrame, plan: Dict[str, Any]) -> str:
        """Format a DataFrame result as a readable response."""
        understanding = plan.get("understanding", "Here are the results")

        if len(df) <= 15:
            table_str = df.to_string(index=False)
            return f"{understanding}\n\n```\n{table_str}\n```"
        else:
            table_str = df.head(15).to_string(index=False)
            return f"{understanding}\n\n```\n{table_str}\n```\n\n(Showing first 15 of {len(df)} rows)"

    def _format_numeric_response(
        self, value: Union[int, float], plan: Dict[str, Any]
    ) -> str:
        """Format a numeric result as a readable response."""
        understanding = plan.get("understanding", "The result is")

        if isinstance(value, float):
            formatted = f"{value:,.2f}"
        else:
            formatted = f"{value:,}"

        return f"{understanding}: **{formatted}**"

    def generate_natural_response(
        self, result: Dict[str, Any], question: str, plan: Dict[str, Any]
    ) -> str:
        """Generate a natural language response from the analysis results."""
        if not result.get("success"):
            return result.get("response", "I couldn't complete the analysis.")

        return result.get("response", "Analysis complete.")
