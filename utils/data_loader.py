import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Set maximum file size
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded file for type and size constraints.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file extension
    file_name = uploaded_file.name.lower()
    valid_extension = any(file_name.endswith(ext) for ext in ALLOWED_EXTENSIONS)

    if not valid_extension:
        logger.warning(f"Invalid file type attempt: {file_name}")
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    # Check file size
    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        logger.warning(f"File too large: {file_name} ({size_mb:.1f}MB)")
        return False, f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB"

    return True, ""


def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load data from uploaded CSV or XLSX file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (DataFrame or None, error_message)
    """
    try:
        file_name = uploaded_file.name.lower()
        logger.info(f"Loading data from file: {file_name}")

        if file_name.endswith(".csv"):
            # Try different encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"Failed to decode CSV: {file_name}")
                return (
                    None,
                    "Could not decode CSV file. Please check the file encoding.",
                )

        elif file_name.endswith((".xlsx", ".xls")):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            logger.info(f"Successfully loaded Excel file: {file_name}")
        else:
            return None, "Unsupported file format"

        # Basic validation
        if df.empty:
            logger.warning(f"Uploaded file is empty: {file_name}")
            return None, "The uploaded file contains no data"

        if len(df.columns) == 0:
            logger.warning(f"Uploaded file has no columns: {file_name}")
            return None, "The uploaded file has no columns"

        return df, ""

    except Exception as e:
        logger.error(f"Error loading file {uploaded_file.name}: {e}", exc_info=True)
        return None, f"Error loading file: {str(e)}"


def get_schema_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract schema information from DataFrame for the Planner agent.

    Args:
        df: pandas DataFrame

    Returns:
        Dictionary with schema information
    """
    schema = {"columns": [], "row_count": len(df), "column_count": len(df.columns)}

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }

        # Add sample values for better understanding
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            if df[col].dtype in ["int64", "float64"]:
                col_info["min"] = float(non_null_values.min())
                col_info["max"] = float(non_null_values.max())
                col_info["mean"] = float(non_null_values.mean())
            else:
                # Sample unique values (up to 5)
                unique_vals = non_null_values.unique()[:5].tolist()
                col_info["sample_values"] = [str(v) for v in unique_vals]

        schema["columns"].append(col_info)

    return schema


def format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """
    Format schema information as a string for LLM prompts.

    Args:
        schema: Schema dictionary from get_schema_info

    Returns:
        Formatted string representation
    """
    lines = [
        f"Dataset: {schema['row_count']} rows x {schema['column_count']} columns",
        "",
        "Columns:",
    ]

    for col in schema["columns"]:
        line = f"  - {col['name']} ({col['dtype']})"

        if "min" in col:
            line += f" [range: {col['min']:.2f} to {col['max']:.2f}]"
        elif "sample_values" in col:
            samples = ", ".join(col["sample_values"][:3])
            line += f" [examples: {samples}]"

        lines.append(line)

    return "\n".join(lines)
