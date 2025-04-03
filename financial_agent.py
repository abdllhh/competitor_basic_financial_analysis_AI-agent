import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from io import StringIO
import requests
import yfinance as yf
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import time

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# HuggingFace API call with retry logic
def query_huggingface(prompt, retries=3, delay=5):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 2000}}
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result[0].get('generated_text', 'No analysis returned') if result else 'Empty response'
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return f"Error: {str(e)} after {retries} attempts"

# Fetch financial data using yfinance
def fetch_yfinance_data(company):
    ticker_map = {
        "Google": "GOOGL",  
        "Microsoft": "MSFT",
        "Meta": "META"
    }
    ticker_symbol = ticker_map.get(company.capitalize(), company.upper())
    try:
        ticker = yf.Ticker(ticker_symbol)
        income_stmt = ticker.quarterly_financials.T
        if income_stmt.empty:
            return f"No financial data available for {company} ({ticker_symbol})"
        
        data = []
        for date in income_stmt.index:
            if "202" in str(date):  # Filter for years starting with "202" (2020-2024)
                revenue = income_stmt.loc[date, "Total Revenue"] if "Total Revenue" in income_stmt.columns else "N/A"
                profit = income_stmt.loc[date, "Net Income"] if "Net Income" in income_stmt.columns else "N/A"
                data.append(f"{date}: Revenue: {revenue:,}, Net Income: {profit:,}")
        return "\n".join(data) if data else f"No data found for {company} ({ticker_symbol})"
    except Exception as e:
        return f"Error fetching data for {company}: {str(e)}"

# State definition
class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Prompts
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather and organize the financial data for MyAICo.AI from the provided CSV data. Provide detailed financial data in a clear table format."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data for MyAICo.AI and provide detailed insights on revenue trends, profitability, and ROI. Focus on the company's performance over time."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of MyAICo.AI with its competitors (Google, Microsoft, Meta) based on the provided data. Use key metrics like revenue, profit margin, and ROI where available. Do not use example data (like CompA or CompB); only use the actual competitor data provided below. If competitor data is missing or limited, note this and proceed with available information. **INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. If the comparison contains errors or missing data, note this specifically and suggest improvements."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report for MyAICo.AI based on the analysis, competitor research, comparison, and feedback provided. Structure it with sections: 1. Executive Summary, 2. Financial Analysis, 3. Competitor Comparison, 4. Recommendations. If any section is missing or contains errors, note it and proceed with available data."""

# Node functions
def gather_financials_node(state: AgentState):
    try:
        df = pd.read_csv(StringIO(state["csv_file"]))
        financial_data_str = df.to_string(index=False)
        prompt = f"{GATHER_FINANCIALS_PROMPT}\n\n{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
        response = query_huggingface(prompt)
        return {"financial_data": response}
    except Exception as e:
        return {"financial_data": f"Error processing CSV data: {str(e)}"}

def analyze_data_node(state: AgentState):
    prompt = f"{ANALYZE_DATA_PROMPT}\n\n{state['financial_data']}"
    response = query_huggingface(prompt)
    return {"analysis": response}

def research_competitors_node(state: AgentState):
    content = state["content"] or []
    for competitor in state["competitors"]:
        result = fetch_yfinance_data(competitor)
        content.append(f"{competitor}:\n{result}")
    return {"content": content}


def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or ["No competitor data available"])
    competitor_names = ", ".join(state["competitors"])  # e.g., "Google, Meta"
    dynamic_prompt = f"""You are an expert financial analyst. Compare the financial performance of MyAICo.AI with its competitors ({competitor_names}) based on the provided data. Use key metrics like revenue, profit margin, and ROI where available. Do not use example data (like CompA or CompB); only use the actual competitor data provided below. Do not hallucinate or assume data not provided. Ensure comparisons account for scale differences (e.g., millions vs. billions). If competitor data is missing or limited, note this and proceed with available information. **INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
    prompt = f"{dynamic_prompt}\n\n{state['task']}\n\nFinancial analysis:\n{state['analysis']}\n\nCompetitor data:\n{content}"
    response = query_huggingface(prompt)
    return {
        "comparison": response,
        "revision_number": state.get("revision_number", 0) + 1
    }

def collect_feedback_node(state: AgentState):
    comparison = state.get("comparison", "No comparison data available")
    prompt = f"{FEEDBACK_PROMPT}\n\n{comparison}"
    response = query_huggingface(prompt)
    return {"feedback": response}

def write_report_node(state: AgentState):
    comparison = state.get("comparison", "No comparison data available")
    feedback = state.get("feedback", "No feedback available")
    prompt = f"{WRITE_REPORT_PROMPT}\n\nComparison: {comparison}\nFeedback: {feedback}"
    response = query_huggingface(prompt)
    return {"report": response}

def should_continue(state):
    return END if state["revision_number"] >= state["max_revisions"] else "collect_feedback"

# Build graph
builder = StateGraph(AgentState)
builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("write_report", write_report_node)

builder.set_entry_point("gather_financials")
builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "research_competitors")
builder.add_edge("research_competitors", "compare_performance")
builder.add_conditional_edges("compare_performance", should_continue)
builder.add_edge("collect_feedback", "compare_performance")
builder.add_edge("compare_performance", "write_report")

graph = builder.compile()

# Streamlit UI
def main():
    st.title("Financial Performance Reporting Agent")
    
    if not HUGGINGFACE_API_KEY:
        st.error("Please configure HUGGINGFACE_API_KEY in .env file")
        return

    task = st.text_input("Enter the task:")
    competitors = st.text_area("Enter competitor names (one per line):").split("\n")
    max_revisions = st.number_input("Max Revisions", min_value=1)
    uploaded_file = st.file_uploader("Upload financial data CSV", type=["csv"])

    if st.button("Start Analysis") and uploaded_file:
        csv_data = uploaded_file.getvalue().decode("utf-8")
        initial_state = {
            "task": task,
            "competitors": [comp.strip() for comp in competitors if comp.strip()],
            "csv_file": csv_data,
            "max_revisions": max_revisions,
            "revision_number": 0,
            "content": []
        }

        with st.spinner("Generating report..."):
            final_state = None
            for step in graph.stream(initial_state):
                st.write(step)
                final_state = list(step.values())[0]  # Extract the state from the step dictionary

            if final_state and "report" in final_state:
                st.subheader("Final Report")
                st.markdown(final_state["report"])
                st.download_button(
                    "Download Report",
                    final_state["report"],
                    file_name="financial_report.md",
                    mime="text/markdown"
                )
            else:
                st.error("Failed to generate report. Check the logs above for details.")

if __name__ == "__main__":
    main()
