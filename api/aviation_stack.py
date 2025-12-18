"""
AviationStack API wrapper for fetching real-time flight data

Free tier: 500 requests/month (~16/day)
Docs: https://aviationstack.com/documentation
"""

import requests
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AviationStackAPI:
    """
    Fetches real-time flight data from AviationStack API
    """
    
    BASE_URL = "http://api.aviationstack.com/v1"
    
    # Major US airports to track
    TRACKED_AIRPORTS = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO"]
    
    def __init__(self):
        self.api_key = os.getenv("AVIATIONSTACK_API_KEY")
        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning("AVIATIONSTACK_API_KEY not set. Using demo mode.")
            self.demo_mode = True
        else:
            self.demo_mode = False
        
        self.requests_today = 0
        self.max_daily_requests = 15
        self.last_request_time = None
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        
        if self.demo_mode:
            return self._get_demo_data(endpoint, params)
        
        # Rate limiting: max 1 request per 2 seconds
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < 2:
                time.sleep(2 - elapsed)
        
        # Check daily limit
        if self.requests_today >= self.max_daily_requests:
            logger.warning("Daily API limit reached. Using demo data.")
            return self._get_demo_data(endpoint, params)
        
        params["access_key"] = self.api_key
        
        try:
            response = requests.get(f"{self.BASE_URL}/{endpoint}", params=params)
            self.requests_today += 1
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API Error: {response.status_code}")
                return self._get_demo_data(endpoint, params)
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return self._get_demo_data(endpoint, params)
    
    def _get_demo_data(self, endpoint: str, params: Dict) -> Dict:
        """Return demo data when API is unavailable"""
        
        airport = params.get("dep_iata", "JFK")
        
        return {
            "data": [
                {
                    "flight": {"iata": "AA123"},
                    "airline": {"name": "American Airlines"},
                    "departure": {
                        "airport": airport,
                        "scheduled": datetime.now().isoformat(),
                        "delay": 15
                    },
                    "arrival": {
                        "airport": "LAX",
                        "scheduled": datetime.now().isoformat()
                    }
                },
                {
                    "flight": {"iata": "DL456"},
                    "airline": {"name": "Delta Air Lines"},
                    "departure": {
                        "airport": airport,
                        "scheduled": datetime.now().isoformat(),
                        "delay": 0
                    },
                    "arrival": {
                        "airport": "ATL",
                        "scheduled": datetime.now().isoformat()
                    }
                },
                {
                    "flight": {"iata": "UA789"},
                    "airline": {"name": "United Airlines"},
                    "departure": {
                        "airport": airport,
                        "scheduled": datetime.now().isoformat(),
                        "delay": 45
                    },
                    "arrival": {
                        "airport": "ORD",
                        "scheduled": datetime.now().isoformat()
                    }
                }
            ]
        }
    
    def get_flights(self, airport: str, flight_type: str = "departure") -> List[Dict]:
        """
        Fetch flights for a specific airport
        
        Args:
            airport: IATA code (e.g., "JFK", "LAX")
            flight_type: "departure" or "arrival"
        
        Returns:
            List of flight dictionaries
        """
        
        params = {
            f"{flight_type[:3]}_iata": airport,
            "limit": 100
        }
        
        data = self._make_request("flights", params)
        return data.get("data", [])
    
    def get_delays_summary(self, airport: str) -> Dict:
        """Get delay summary for an airport"""
        
        flights = self.get_flights(airport, "departure")
        
        total = len(flights)
        if total == 0:
            return {
                "airport": airport,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_flights": 0,
                "delayed_flights": 0,
                "delay_rate": 0,
                "avg_delay": 0
            }
        
        delayed = sum(1 for f in flights 
                     if f.get("departure", {}).get("delay", 0) and 
                     f.get("departure", {}).get("delay", 0) > 15)
        
        total_delay = sum(f.get("departure", {}).get("delay", 0) or 0 
                         for f in flights)
        
        return {
            "airport": airport,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_flights": total,
            "delayed_flights": delayed,
            "delay_rate": round(delayed / total, 3) if total > 0 else 0,
            "avg_delay": round(total_delay / total, 1) if total > 0 else 0
        }
    
    def get_all_airports_summary(self) -> List[Dict]:
        """Get delay summary for all tracked airports"""
        
        summaries = []
        for airport in self.TRACKED_AIRPORTS:
            summary = self.get_delays_summary(airport)
            summaries.append(summary)
            logger.info(f"{airport}: {summary['delay_rate']:.1%} delays")
        
        return summaries


# Quick test
if __name__ == "__main__":
    api = AviationStackAPI()
    print("\n📊 Testing AviationStack API...\n")
    
    summary = api.get_delays_summary("JFK")
    print(f"Airport: {summary['airport']}")
    print(f"Total Flights: {summary['total_flights']}")
    print(f"Delayed: {summary['delayed_flights']}")
    print(f"Delay Rate: {summary['delay_rate']:.1%}")

