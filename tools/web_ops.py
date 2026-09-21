"""
JARVIS Assistant - Web & Information Retrieval Tools
Provides web search, weather data, datetime resolution, and browser navigation
callable by Gemini function calling:
- Web search via DuckDuckGo Instant Answer API
- Current weather via wttr.in API
- Date and time lookup (local and timezones)
- Browser website navigation
"""

from datetime import datetime
import logging
import re
import urllib.parse
import webbrowser
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

logger = logging.getLogger(__name__)

# Request timeout in seconds for web services
REQUEST_TIMEOUT: int = 10

# Standard user agent for HTTP requests
USER_AGENT: str = "JARVIS-Assistant/1.0 (Windows NT 10.0; Win64; x64)"

# Common timezone abbreviations mapped to IANA timezone names
TIMEZONE_ALIASES: dict[str, str] = {
    "utc": "UTC",
    "gmt": "GMT",
    "est": "America/New_York",
    "edt": "America/New_York",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "ist": "Asia/Kolkata",
    "bst": "Europe/London",
    "cet": "Europe/Paris",
    "cest": "Europe/Paris",
    "jst": "Asia/Tokyo",
    "aest": "Australia/Sydney",
}


try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def web_search(query: str) -> str:
    """
    Search the web for information using real-time web search.

    Args:
        query: The search term or question to query.

    Returns:
        Summary of top search results (titles and snippets), or an informative message.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    search_query = query.strip()
    logger.info("Performing web search for: '%s'", search_query)

    # 1. Primary: Use real web search via DDGS
    if DDGS is not None:
        try:
            results = list(DDGS().text(search_query, max_results=3))
            if results:
                snippets = []
                for r in results:
                    title = r.get("title", "").strip()
                    body = r.get("body", "").strip()
                    if title and body:
                        snippets.append(f"{title}: {body}")
                    elif body:
                        snippets.append(body)
                if snippets:
                    logger.info("Found %d web results for '%s'", len(snippets), search_query)
                    return "\n".join(snippets)
        except Exception as e:
            logger.warning("DDGS web search error for '%s': %s", search_query, e)

    # 2. Fallback: DuckDuckGo Instant Answer API
    try:
        encoded_query = urllib.parse.quote_plus(search_query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=3.5,
        )
        if response.ok:
            data = response.json()
            answer = data.get("Answer") or data.get("AbstractText") or data.get("Abstract")
            if answer:
                logger.info("Found instant answer fallback for '%s'", search_query)
                return str(answer)
    except Exception as e:
        logger.debug("Instant answer fallback error: %s", e)

    return f"No search results found for '{search_query}'. You can ask me to open a search in your browser."


def get_weather(city: str) -> str:
    """
    Retrieve current weather information for a specified city using wttr.in.

    Args:
        city: Name of the city or location (e.g. 'London', 'New York', 'Tokyo').

    Returns:
        A formatted string describing the current temperature, weather condition,
        humidity, and feels-like temperature.
    """
    if not city or not city.strip():
        return "Error: City name cannot be empty."

    city_name = city.strip()
    encoded_city = urllib.parse.quote_plus(city_name)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    logger.info("Fetching weather for '%s'", city_name)

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        current_list = data.get("current_condition", [])
        if not current_list:
            logger.warning("No current_condition found in weather response for '%s'", city_name)
            return f"Weather data not available for '{city_name}'."

        current = current_list[0]
        temp_c = current.get("temp_C", "N/A")
        temp_f = current.get("temp_F", "N/A")
        feels_like_c = current.get("FeelsLikeC", "N/A")
        feels_like_f = current.get("FeelsLikeF", "N/A")
        humidity = current.get("humidity", "N/A")

        # Condition description
        weather_descs = current.get("weatherDesc", [])
        condition = weather_descs[0].get("value", "Unknown").strip() if weather_descs else "Unknown"

        # Wind speed
        wind_kmph = current.get("windspeedKmph", "N/A")

        # Resolved location name
        area_list = data.get("nearest_area", [])
        if area_list:
            area_name = area_list[0].get("areaName", [{}])[0].get("value", city_name)
            country = area_list[0].get("country", [{}])[0].get("value", "")
            location = f"{area_name}, {country}" if country else area_name
        else:
            location = city_name.title()

        logger.info("Successfully fetched weather for '%s'", location)
        return (
            f"Current weather in {location}:\n"
            f"- Condition: {condition}\n"
            f"- Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"- Feels like: {feels_like_c}°C ({feels_like_f}°F)\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind Speed: {wind_kmph} km/h"
        )

    except requests.Timeout:
        logger.error("Weather request timed out for city: '%s'", city_name)
        return f"Error: Weather request timed out after {REQUEST_TIMEOUT} seconds for '{city_name}'."
    except requests.ConnectionError as e:
        logger.error("Connection error during weather fetch for '%s': %s", city_name, e)
        return "Error: Could not connect to the weather service. Please check your internet connection."
    except requests.RequestException as e:
        logger.error("HTTP error during weather fetch for '%s': %s", city_name, e)
        return f"Error retrieving weather for '{city_name}': {str(e)}"
    except Exception as e:
        logger.error("Unexpected error parsing weather for '%s': %s", city_name, e, exc_info=True)
        return f"Error processing weather data: {str(e)}"


def get_time(timezone: str = "local") -> str:
    """
    Get the current date and time for a given timezone or local system time.

    Args:
        timezone: Target timezone name. Defaults to 'local' for host system time.
                  Supports common aliases (e.g. 'UTC', 'GMT', 'EST', 'PST', 'IST')
                  and standard IANA timezone names (e.g. 'America/New_York', 'Asia/Kolkata').

    Returns:
        A formatted string with the day of the week, full date, time, and timezone.
    """
    tz_input = timezone.strip() if timezone else "local"
    tz_lower = tz_input.lower()

    logger.info("Resolving current time for timezone: '%s'", tz_input)

    try:
        if tz_lower in ("local", "", "default"):
            now = datetime.now().astimezone()
            tz_label = "Local Time"
        else:
            iana_name = TIMEZONE_ALIASES.get(tz_lower, tz_input)
            tz_obj = ZoneInfo(iana_name)
            now = datetime.now(tz_obj)
            tz_label = iana_name

        date_str = now.strftime("%A, %B %d, %Y")
        time_str = now.strftime("%I:%M:%S %p")
        tz_abbr = now.strftime("%Z") or tz_label

        return f"Current date and time ({tz_label}): {date_str}, {time_str} ({tz_abbr})"

    except ZoneInfoNotFoundError:
        logger.warning("Unrecognized timezone '%s', falling back to local time", tz_input)
        local_now = datetime.now().astimezone()
        date_str = local_now.strftime("%A, %B %d, %Y")
        time_str = local_now.strftime("%I:%M:%S %p")
        tz_abbr = local_now.strftime("%Z") or "Local"
        return (
            f"Unknown timezone '{tz_input}'. Current local time: {date_str}, "
            f"{time_str} ({tz_abbr})"
        )
    except Exception as e:
        logger.error("Error retrieving current time: %s", e, exc_info=True)
        return f"Error retrieving current time: {str(e)}"


def open_website(url: str) -> str:
    """
    Open a website URL in the default web browser.

    Args:
        url: The web URL or domain name to open (e.g. 'https://google.com' or 'github.com').

    Returns:
        A confirmation message indicating the URL has been opened, or an error description.
    """
    if not url or not url.strip():
        return "Error: URL cannot be empty."

    target_url = url.strip()

    # Prepend https:// if no protocol scheme is specified
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target_url):
        target_url = f"https://{target_url}"

    logger.info("Opening URL in default browser: '%s'", target_url)

    try:
        # new=2 opens in a new browser tab if supported
        opened = webbrowser.open(target_url, new=2)
        if opened:
            logger.info("Browser successfully opened '%s'", target_url)
            return f"Opened {target_url} in the default browser."
        else:
            logger.warning("webbrowser.open returned False for '%s'", target_url)
            return f"Attempted to open {target_url}, but the browser did not confirm launch."
    except Exception as e:
        logger.error("Failed to open URL '%s': %s", target_url, e, exc_info=True)
        return f"Failed to open website '{url}': {str(e)}"


def search_youtube(query: str) -> str:
    """
    Search for videos on YouTube and open the search results in the default browser.

    Args:
        query: The video search terms, title, artist, or topic to look up.

    Returns:
        Confirmation that YouTube search has been opened.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    encoded = urllib.parse.quote_plus(query.strip())
    url = f"https://www.youtube.com/results?search_query={encoded}"
    logger.info("Opening YouTube search for '%s': %s", query, url)
    return open_website(url)


def search_google(query: str) -> str:
    """
    Perform a Google search and open the search results in the default browser.

    Args:
        query: The query or keywords to search on Google.

    Returns:
        Confirmation that Google search has been opened.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    encoded = urllib.parse.quote_plus(query.strip())
    url = f"https://www.google.com/search?q={encoded}"
    logger.info("Opening Google search for '%s': %s", query, url)
    return open_website(url)
