from typing import Any, Dict
import os

import ast
import json

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ---------------------------
# Initialization
# ---------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # unchanged


# ---------------------------
# Data loading
# ---------------------------
df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")
df = pd.concat([df_train, df_test])

categorical_features = [
    "ExterQual",
    "BsmtQual",
    "HeatingQC",
    "KitchenQual",
    "Neighborhood",
    "Foundation",
    "BsmtFinType1",
    "GarageType",
    "GarageFinish",
    "GarageFinish",
    "OverallQual",
    "GrLivArea"
]

categories: Dict[str, Any] = {}

for feature in categorical_features:
    categories[feature] = df[feature].sort_values().unique()


# ---------------------------
# Prompts
# ---------------------------
main_prompt = """
Extract features from the text enclosed between triple backquotes.
The output should be a python dictionary and the keys should not change.
If the value of a feature is specified in the text enclosed by triple backticks:
Substitute the word 'None' in the features dictionary by its value from the text.
Else, keep it as None.

In the end return the new modified features dictionary.
Return ONLY a valid JSON object with double quotes, no extra text.

The features dictionary is:
"""

features = """
{
"LotArea" : "None",
"SaleCondition" : "None",
"OverallQual" : "None",
"BsmtQual" : "None",
"YearBuilt" : "None",
"ExterQual" : "None",
"GrLivArea" : "None",
"TotalBsmtSF" : "None",
"GarageCars" : "None",
"KitchenQual" : "None",
"Neighborhood" : "None",
"Condition1" : "None",
}
"""

categories_values = """
For the following features, they can only have the specified values/categories.
If you are not able to infer the value or category of these features from the text in triple backquotes,
then keep the value None in the returned dictionary
else, fill it with the allowed categories.
It is important to keep in mind that for infering, the user might not give direct answer to fill,
so infer is important.
For the other features not specified in the following list,
they have values as numbers only and no categories or text allowed.
"""


# ---------------------------
# Core function
# ---------------------------
def extract_features(query: str, json_feats2: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured features from user text using LLM."""
    full_prompt = (
        f"{main_prompt}\n{json_feats2}\n{categories_values}\n{categories}\n```{query}```"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.1,
    )

    raw_output = response.choices[0].message.content

    try:
        parsed_dict: Dict[str, Any] = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed_dict = ast.literal_eval(raw_output)

    return parsed_dict
