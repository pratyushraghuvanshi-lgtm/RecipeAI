import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Verify API key exists
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

# Initialize the modern Gemini Client
client = genai.Client(api_key=api_key)

app = FastAPI()

# Define the data structure expected from the frontend
class RecipeRequest(BaseModel):
    ingredients: list[str]
    diet: str
    time: str
    meal: str

# 1. API Endpoint for generating recipes
@app.post("/api/recipes")
async def generate_recipes(request: RecipeRequest):
    if not request.ingredients:
        raise HTTPException(status_code=400, detail="Ingredients are required")

    # Format preferences cleanly for the AI prompt
    diet_pref = request.diet if request.diet else "Any"
    time_pref = request.time if request.time else "Any"
    meal_pref = request.meal if request.meal else "Any"
    ingredients_list = ", ".join(request.ingredients)

    prompt = f"""
    You are an expert chef AI. The user has these ingredients in their pantry: {ingredients_list}.
    Their preferences are -> Diet: {diet_pref}, Max Time: {time_pref}, Meal Type: {meal_pref}.
    
    Generate up to 3 recipes using mostly these ingredients. You can assume they have basic pantry staples (salt, pepper, oil, water).
    
    Respond STRICTLY in JSON format matching this exact schema:
    {{
      "recipes": [
        {{
          "name": "Recipe Name",
          "time": "Total time (e.g., 20 min)",
          "difficulty": "Easy/Medium/Hard",
          "servings": "Number (e.g., 2)",
          "match": "Uses X/Y ingredients",
          "description": "Short appetizing description",
          "ingredients": ["quantity and ingredient 1", "quantity and ingredient 2"],
          "steps": ["Step 1", "Step 2"],
          "tip": "A helpful cooking tip"
        }}
      ]
    }}
    """

    try:
        # Request generation using the standard gemini-2.5-flash model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        
        # Parse the JSON string from Gemini back into a structured dictionary
        recipe_data = json.loads(response.text)
        return recipe_data
        
    except Exception as e:
        print(f"Error generating recipes: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recipes")

# 2. Serve the frontend assets from the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")