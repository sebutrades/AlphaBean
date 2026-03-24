"""Discover available methods on the Massive RESTClient."""
from massive import RESTClient
import os
from dotenv import load_dotenv
load_dotenv()

client = RESTClient(os.getenv("MASSIVE_API_KEY"))

# Print all public methods
methods = [m for m in dir(client) if not m.startswith("_")]
print(f"RESTClient has {len(methods)} public methods/attrs:\n")
for m in sorted(methods):
    print(f"  {m}")