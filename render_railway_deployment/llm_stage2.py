from typing import Any
import os

from dotenv import load_dotenv
from groq import Groq


# ---------------------------
# Initialization
# ---------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # unchanged


# ---------------------------
# Core function
# ---------------------------
def interpret_prediction(features: Any, price: float) -> str:
    """Generate explanation for predicted price."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f" Given {features}, explain in 1 small paragraph the reson for this price {price}",
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content
