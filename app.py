import pandas as pd
import gradio as gr
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage

def infer_column_meanings(columns):
    chat = ChatVertexAI(model_name="gemini-pro")
    prompt = f"""
    You are an expert in NBA statistics and data analysis. Given the following column names from an NBA dataset, infer their meanings:
    {', '.join(columns)}
    
    Provide a dictionary where each column name is a key, and the value is a concise explanation of what the column represents.
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    
    response = chat.invoke(messages)
    return response.content

def analyze_csv(file):
    df = pd.read_csv(file.name, nrows=0)  # Solo leer los headers
    columns = list(df.columns)
    meanings = infer_column_meanings(columns)
    return meanings

# Interfaz Gradio
demo = gr.Interface(
    fn=analyze_csv,
    inputs=gr.File(),
    outputs=gr.Textbox(),
    title="NBA Column Meaning Inference",
    description="Upload a CSV file to infer the meanings of its column names."
)

demo.launch(share=True, server_name="localhost", server_port=8002)

