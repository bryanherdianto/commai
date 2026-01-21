import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chart color scheme
CHART_COLORS = [
    "#96e6a1",  # Light Green
    "#d4fc79",  # Chartreuse
    "#22C55E",  # Green
    "#14B8A6",  # Teal
    "#059669",  # Emerald
    "#34D399",  # Medium Green
    "#A7F3D0",  # Pale Green
    "#065F46",  # Dark Green
]

# Specific colors for Sales and Profit
METRIC_COLORS = {
    "Sales": "#96e6a1",  # Light Green
    "Profit": "#22C55E",  # Success Green
}


def create_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Optional[go.Figure]:
    """
    Create a Plotly chart based on the specified type.

    Args:
        df: pandas DataFrame with the data
        chart_type: Type of chart ('bar', 'line', 'pie', 'scatter', 'hbar', 'area')
        x: Column name for x-axis
        y: Column name for y-axis (can be a list for grouped bar)
        color: Column name for color grouping
        title: Chart title
        **kwargs: Additional arguments for the chart

    Returns:
        Plotly Figure object or None if creation fails
    """
    logger.info(f"Creating chart: {chart_type} (x={x}, y={y})")
    try:
        chart_type = chart_type.lower().strip()

        # Handle map type with geocoding
        if chart_type in ["map", "geo", "choropleth", "bubble_map"]:
            return _create_map_chart(df, x, y, title, **kwargs)

        # Check if we have multiple numeric columns for grouped bar chart
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        is_multi_metric = len(numeric_cols) >= 2 and x and x not in numeric_cols

        # Common styling options
        common_opts = {
            "color_discrete_sequence": CHART_COLORS,
            "template": "plotly_dark",
        }

        if title:
            common_opts["title"] = title

        if chart_type in ["bar", "vertical_bar"]:
            if is_multi_metric and not y:
                # Grouped bar chart for multiple metrics (e.g., Sales and Profit)
                fig = _create_grouped_bar_chart(df, x, numeric_cols, title)
            else:
                fig = px.bar(df, x=x, y=y, color=color, **common_opts)

        elif chart_type in ["hbar", "horizontal_bar", "barh"]:
            fig = px.bar(df, x=y, y=x, color=color, orientation="h", **common_opts)

        elif chart_type in ["line", "trend"]:
            fig = px.line(df, x=x, y=y, color=color, markers=True, **common_opts)
            # Fix x-axis for year data - force integer ticks
            if x and "year" in x.lower():
                fig.update_xaxes(dtick=1, tickformat="d")

        elif chart_type in ["pie", "donut"]:
            values_col = (
                y
                if y
                else (
                    kwargs.get("values")
                    or df.select_dtypes(include=["number"]).columns[0]
                )
            )
            names_col = (
                x
                if x
                else (
                    kwargs.get("names")
                    or df.select_dtypes(include=["object"]).columns[0]
                )
            )

            fig = px.pie(
                df,
                values=values_col,
                names=names_col,
                color_discrete_sequence=CHART_COLORS,
                template="plotly_dark",
                title=title,
                hole=0.4 if chart_type == "donut" else 0,
            )

        elif chart_type == "scatter":
            # Ensure we have x and y for scatter
            if not x or not y:
                # Auto-detect numeric columns for scatter
                if len(numeric_cols) >= 2:
                    x = x or numeric_cols[0]
                    y = y or numeric_cols[1]

            # Try with trendline first, fall back to without if statsmodels not available
            try:
                fig = px.scatter(
                    df, x=x, y=y, color=color, trendline="ols", **common_opts
                )
            except Exception:
                # Fallback without trendline
                fig = px.scatter(df, x=x, y=y, color=color, **common_opts)

        elif chart_type == "area":
            fig = px.area(df, x=x, y=y, color=color, **common_opts)

        elif chart_type in ["histogram", "hist"]:
            fig = px.histogram(df, x=x, color=color, **common_opts)

        elif chart_type in ["box", "boxplot"]:
            fig = px.box(df, x=x, y=y, color=color, **common_opts)

        elif chart_type in ["count", "count_plot", "countplot"]:
            if x:
                count_df = df[x].value_counts().reset_index()
                count_df.columns = [x, "Count"]
                fig = px.bar(count_df, x=x, y="Count", **common_opts)
            else:
                return None

        elif chart_type in ["multi_line", "multiline"]:
            fig = px.line(df, x=x, y=y, color=color, markers=True, **common_opts)

        else:
            # Default to bar chart
            if is_multi_metric:
                fig = _create_grouped_bar_chart(df, x, numeric_cols, title)
            else:
                fig = px.bar(df, x=x, y=y, color=color, **common_opts)

        # Apply dark theme styling
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E5E7EB", family="Inter, sans-serif"),
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"
            ),
        )

        return fig

    except KeyError as e:
        logger.error(f"Column not found: {e}")
        return _create_unsupported_chart_message(
            f"Data error: Column {str(e)} missing."
        )
    except ValueError as e:
        logger.error(f"Value error in plotting: {e}")
        return _create_unsupported_chart_message(
            "Visualisation error: Incompatible data types."
        )
    except Exception as e:
        logger.error(f"Unexpected error creating chart: {e}", exc_info=True)
        return None


def _create_grouped_bar_chart(
    df: pd.DataFrame, x_col: str, y_cols: List[str], title: Optional[str] = None
) -> go.Figure:
    """
    Create a grouped bar chart for multiple metrics (e.g., Sales and Profit).
    Uses distinct colors for each metric.
    """
    fig = go.Figure()

    # Use specific colors for known metrics, otherwise use palette
    for i, col in enumerate(y_cols):
        bar_color = METRIC_COLORS.get(col, CHART_COLORS[i % len(CHART_COLORS)])
        fig.add_trace(go.Bar(name=col, x=df[x_col], y=df[col], marker_color=bar_color))

    fig.update_layout(
        barmode="group",
        title=title,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def _create_map_chart(
    df: pd.DataFrame,
    location_col: Optional[str] = None,
    value_col: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """
    Create a map visualization with geocoded locations.

    Args:
        df: DataFrame with location data
        location_col: Column containing location names (city, state, country)
        value_col: Column with values for bubble size/color
        title: Chart title

    Returns:
        Plotly Figure with map
    """
    from utils.geocoding import geocode_dataframe, is_geopy_available

    if not is_geopy_available():
        return _create_unsupported_chart_message(
            "Map visualization requires 'geopy' library. Install with: pip install geopy"
        )

    # Auto-detect location column if not provided
    if not location_col:
        location_candidates = [
            "city",
            "state",
            "country",
            "region",
            "location",
            "place",
        ]
        for col in df.columns:
            if any(cand in col.lower() for cand in location_candidates):
                location_col = col
                break
        if not location_col:
            location_col = (
                df.select_dtypes(include=["object"]).columns[0]
                if len(df.select_dtypes(include=["object"]).columns) > 0
                else None
            )

    if not location_col or location_col not in df.columns:
        return _create_unsupported_chart_message(
            "No location column found for map visualization."
        )

    # Auto-detect value column if not provided
    if not value_col:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]

    # Check if lat/lon already exist
    if "lat" in df.columns and "lon" in df.columns:
        geo_df = df.dropna(subset=["lat", "lon"])
    else:
        # Detect country column for better geocoding
        country_col = None
        for col in df.columns:
            if "country" in col.lower():
                country_col = col
                break

        # Geocode the dataframe
        try:
            geo_df = geocode_dataframe(df, location_col, country_col, max_locations=30)
        except Exception as e:
            return _create_unsupported_chart_message(f"Geocoding failed: {str(e)}")

    if len(geo_df) == 0:
        return _create_unsupported_chart_message(
            "Could not geocode any locations. Try using more specific location names."
        )

    # Create the map
    if value_col and value_col in geo_df.columns:
        fig = px.scatter_geo(
            geo_df,
            lat="lat",
            lon="lon",
            size=value_col,
            color=value_col,
            hover_name=location_col,
            hover_data=[value_col],
            color_continuous_scale=["#d4fc79", "#96e6a1", "#22C55E"],
            title=title or f"{value_col} by {location_col}",
            template="plotly_dark",
        )
    else:
        fig = px.scatter_geo(
            geo_df,
            lat="lat",
            lon="lon",
            hover_name=location_col,
            title=title or f"Locations: {location_col}",
            template="plotly_dark",
        )

    # Style the map
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.3)",
        showland=True,
        landcolor="rgba(30, 41, 59, 1)",  # Dark slate
        showocean=True,
        oceancolor="rgba(15, 23, 42, 1)",  # Darker blue
        showlakes=True,
        lakecolor="rgba(15, 23, 42, 1)",
        showcountries=True,
        countrycolor="rgba(255,255,255,0.2)",
        projection_type="natural earth",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )

    return fig


def _create_unsupported_chart_message(message: str) -> go.Figure:
    """Create a figure with a message for unsupported chart types."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#F59E0B"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def create_scatter_from_df(
    df: pd.DataFrame, x_col: str, y_col: str, title: Optional[str] = None
) -> go.Figure:
    """
    Create a scatter plot with trend line from raw DataFrame.
    Used for correlation analysis.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        trendline="ols",
        color_discrete_sequence=CHART_COLORS,
        template="plotly_dark",
        title=title,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"
        ),
    )

    return fig


def detect_chart_type(question: str) -> str:
    """
    Detect the likely chart type from the user's question.
    """
    question_lower = question.lower()

    if "pie" in question_lower or "distribution" in question_lower:
        return "pie"
    elif (
        "line" in question_lower
        or "trend" in question_lower
        or "over time" in question_lower
    ):
        return "line"
    elif (
        "scatter" in question_lower
        or "correlation" in question_lower
        or "relationship" in question_lower
    ):
        return "scatter"
    elif "horizontal" in question_lower or "hbar" in question_lower:
        return "hbar"
    elif "count" in question_lower or "how many" in question_lower:
        return "count"
    elif "area" in question_lower:
        return "area"
    elif "box" in question_lower:
        return "box"
    elif (
        "bar" in question_lower
        or "chart" in question_lower
        or "compare" in question_lower
    ):
        return "bar"
    else:
        return "bar"


def format_chart_data(
    df: pd.DataFrame,
    group_by: Optional[str] = None,
    agg_column: Optional[str] = None,
    agg_func: str = "sum",
) -> pd.DataFrame:
    """Format DataFrame for chart creation by aggregating data."""
    if group_by and agg_column:
        if agg_func == "sum":
            result = df.groupby(group_by)[agg_column].sum().reset_index()
        elif agg_func == "mean":
            result = df.groupby(group_by)[agg_column].mean().reset_index()
        elif agg_func == "count":
            result = df.groupby(group_by)[agg_column].count().reset_index()
        elif agg_func == "min":
            result = df.groupby(group_by)[agg_column].min().reset_index()
        elif agg_func == "max":
            result = df.groupby(group_by)[agg_column].max().reset_index()
        else:
            result = df.groupby(group_by)[agg_column].sum().reset_index()
        return result
    return df
