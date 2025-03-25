import os
import yfinance as yf
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv(dotenv_path=".env")
GROQ_API_KEY =os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("Loaded GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Initialize LLMs
groq_llm = ChatGroq(model="Llama-3.3-70b-Versatile", api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_llm = genai.GenerativeModel("gemini-2.0-flash")

# Function to fetch stock price from Yahoo Finance
def fetch_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        current_price = data['Close'].iloc[-1]  # Last closing price
        return round(current_price, 2)
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"

# Prompt template for LLM analysis
def generate_prompt(ticker, price):
    return f"""
    The current stock price of {ticker} is ${price}.
    Perform a financial analysis and determine whether it is a Buy, Hold, or Sell.

    **Fundamental Analysis**:
    - Evaluate financial metrics such as P/E ratio, earnings, revenue growth.
    - Check company news, market trends, and economic impact.

    **Technical Analysis**:
    - Analyze moving averages, RSI, MACD, volume trends, and support/resistance levels.

    Provide a final recommendation: **BUY, SELL, or HOLD**, with reasoning.
    """

# Function to analyze stock using Groq
def analyze_stock_groq(ticker, price):
    try:
        prompt = generate_prompt(ticker, price)
        response = groq_llm.invoke(prompt)  # AIMessage object
        return response.content if hasattr(response, "content") else str(response)  # ✅ Extract text
    except Exception as e:
        print(f"Error in Groq analysis: {e}")
        return "Error in Groq analysis."

# Function to analyze stock using Gemini
def analyze_stock_gemini(ticker, price):
    try:
        prompt = generate_prompt(ticker, price)
        response = gemini_llm.generate_content(prompt)
        
        # Check if response exists and contains text
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "Error: Empty or unexpected response from Gemini."
    
    except Exception as e:
        return f"Error in Gemini analysis: {str(e)}"

# API Route
@app.route('/get_stock', methods=['GET'])
def get_stock():
    ticker = request.args.get('ticker')

    if not ticker:
        return jsonify({"error": "Please provide a stock ticker."}), 400

    # Fetch stock price
    price = fetch_stock_price(ticker.upper())

    if isinstance(price, str):
        return jsonify({"error": price}), 500  # Error fetching price

    # Analyze stock using both LLMs
    groq_analysis = analyze_stock_groq(ticker.upper(), price)
    gemini_analysis = analyze_stock_gemini(ticker.upper(), price)

    return jsonify({
        "ticker": ticker.upper(),
        "price": price,
        "groq_analysis": groq_analysis,  
        "gemini_analysis": gemini_analysis  # ✅ Now JSON serializable
    })

# Run the API
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
