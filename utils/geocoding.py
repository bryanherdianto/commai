import pandas as pd
from typing import Dict, Tuple, Optional
import time
import logging

# Configure logger
logger = logging.getLogger(__name__)

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# Cache for geocoded locations to avoid repeated API calls
_geocode_cache: Dict[str, Tuple[float, float]] = {}

# Common US cities with pre-cached coordinates to speed up lookups
US_CITY_COORDS = {
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611),
    "dallas": (32.7767, -96.7970),
    "san jose": (37.3382, -121.8863),
    "austin": (30.2672, -97.7431),
    "jacksonville": (30.3322, -81.6557),
    "san francisco": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903),
    "boston": (42.3601, -71.0589),
    "miami": (25.7617, -80.1918),
    "atlanta": (33.7490, -84.3880),
    "detroit": (42.3314, -83.0458),
    "minneapolis": (44.9778, -93.2650),
    # Add more as needed
}


def geocode_location(
    location: str, country_hint: str = None, state_hint: str = None
) -> Optional[Tuple[float, float]]:
    """
    Geocode a location string to latitude/longitude coordinates.

    Args:
        location: Location name (city, state, country)
        country_hint: Optional country to help with ambiguous locations
        state_hint: Optional state for US locations

    Returns:
        Tuple of (latitude, longitude) or None if not found
    """
    if not location or pd.isna(location):
        return None

    location = str(location).strip()
    location_lower = location.lower()

    # Check cache first
    cache_key = f"{location_lower}_{state_hint or ''}_{country_hint or ''}"
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    # Check pre-cached US cities
    if location_lower in US_CITY_COORDS:
        return US_CITY_COORDS[location_lower]

    if not GEOPY_AVAILABLE:
        return None

    # Use Nominatim geocoder
    try:
        geolocator = Nominatim(user_agent="intelligent_data_room_app")

        # Build query with state and country for better accuracy
        if state_hint and country_hint:
            query = f"{location}, {state_hint}, {country_hint}"
        elif state_hint:
            query = f"{location}, {state_hint}, USA"
        elif country_hint:
            query = f"{location}, {country_hint}"
        else:
            query = f"{location}, USA"  # Default to USA for ambiguous names

        # Geocode with timeout
        result = geolocator.geocode(query, timeout=5)

        if result:
            coords = (result.latitude, result.longitude)
            _geocode_cache[cache_key] = coords
            return coords

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error for {location}: {e}")
    except Exception as e:
        print(f"Unexpected geocoding error: {e}")

    return None


def geocode_dataframe(
    df: pd.DataFrame,
    location_col: str,
    country_col: str = None,
    state_col: str = None,
    max_locations: int = 50,
) -> pd.DataFrame:
    """
    Add latitude and longitude columns to a DataFrame based on location names.

    Args:
        df: DataFrame with location data
        location_col: Column name containing location names
        country_col: Optional column with country names for better accuracy
        state_col: Optional column with state names for US locations
        max_locations: Maximum number of unique locations to geocode (to prevent API abuse)

    Returns:
        DataFrame with added 'lat' and 'lon' columns
    """
    df = df.copy()

    # Auto-detect state column if not provided
    if not state_col:
        for col in df.columns:
            if "state" in col.lower():
                state_col = col
                break

    # Get unique locations
    unique_locations = df[location_col].dropna().unique()[:max_locations]

    # Build location to coords mapping
    location_coords = {}
    for loc in unique_locations:
        country = None
        state = None

        loc_rows = df[df[location_col] == loc]

        if country_col and country_col in df.columns:
            country_matches = loc_rows[country_col].mode()
            if len(country_matches) > 0:
                country = country_matches.iloc[0]

        if state_col and state_col in df.columns:
            state_matches = loc_rows[state_col].mode()
            if len(state_matches) > 0:
                state = state_matches.iloc[0]

        coords = geocode_location(loc, country, state)
        if coords:
            location_coords[loc] = coords

        # Small delay to respect rate limits
        time.sleep(0.1)

    # Apply to dataframe
    df["lat"] = df[location_col].map(
        lambda x: location_coords.get(x, (None, None))[0]
        if x in location_coords
        else None
    )
    df["lon"] = df[location_col].map(
        lambda x: location_coords.get(x, (None, None))[1]
        if x in location_coords
        else None
    )

    # Remove rows without coordinates
    df = df.dropna(subset=["lat", "lon"])

    return df


def is_geopy_available() -> bool:
    """Check if geopy is available for geocoding."""
    return GEOPY_AVAILABLE
