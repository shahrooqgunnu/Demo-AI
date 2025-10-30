import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
import plotly.express as px
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from datetime import datetime, timedelta
 
# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
 
# -----------------------------
# Initialize OpenAI LLM
# -----------------------------
if not openai_api_key:
    st.error("⚠️ Please set your OPENAI_API_KEY in a .env file.")
else:
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.7)
 
# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Agentic AI for Smart Cloud Infrastructure", layout="wide")
st.title("🤖 Agentic AI for Smart Cloud Infrastructure Management")
st.markdown("---")
 
# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio("Navigation", ["Dashboard", "Cost Trends", "AI Assistant", "About"])
 
# -----------------------------
# Helper: Fetch AWS cost data
# -----------------------------
@st.cache_data
def get_aws_cost_data():
    try:
        ce = boto3.client("ce")
        end = datetime.today().date()
        start = end - timedelta(days=30)
        response = ce.get_cost_and_usage(
            TimePeriod={"Start": str(start), "End": str(end)},
            Granularity="DAILY",
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        )
        data = []
        for group in response["ResultsByTime"]:
            for g in group["Groups"]:
                service = g["Keys"][0]
                cost = float(g["Metrics"]["UnblendedCost"]["Amount"])
                data.append({"Service": service, "Cost ($)": round(cost, 2)})
        df = pd.DataFrame(data)
        df = df.groupby("Service").sum().reset_index().sort_values("Cost ($)", ascending=False)
        return df
    except (NoCredentialsError, PartialCredentialsError):
        st.warning("⚠️ AWS credentials not found. Showing demo data.")
        data = {
            "Service": ["EC2", "S3", "EKS", "RDS", "Lambda"],
            "Cost ($)": [3200, 1100, 850, 1900, 400],
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching AWS data: {e}")
        return pd.DataFrame()
 
# -----------------------------
# Dashboard Page
# -----------------------------
if menu == "Dashboard":
    st.header("🌩️ Cloud Cost Overview")
 
    df = get_aws_cost_data()
 
    if df.empty:
        st.stop()
 
    col1, col2 = st.columns(2)
 
    with col1:
        fig1 = px.bar(df, x="Service", y="Cost ($)", color="Service", title="💰 Monthly Cloud Cost by Service")
        st.plotly_chart(fig1, use_container_width=True)
 
    with col2:
        df["Idle Resource (%)"] = [25, 10, 15, 20, 5][: len(df)]
        fig2 = px.pie(df, names="Service", values="Idle Resource (%)", title="🧠 Idle Resource Distribution")
        st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown("### 🔍 Key Insights")
    st.info("""
    - **EC2** contributes the highest cost — optimization recommended.  
    - **RDS** has moderate usage but potential idle time.  
    - **S3** utilization is efficient.  
    """)
 
# -----------------------------
# Cost Trends Page
# -----------------------------
elif menu == "Cost Trends":
    st.header("📊 Cloud Cost & Usage Trends")
 
    try:
        ce = boto3.client("ce")
        end = datetime.today().date()
        start = end - timedelta(days=150)
        response = ce.get_cost_and_usage(
            TimePeriod={"Start": str(start), "End": str(end)},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
        )
        months = []
        totals = []
        for r in response["ResultsByTime"]:
            month = r["TimePeriod"]["Start"]
            total = float(r["Total"]["UnblendedCost"]["Amount"])
            months.append(month)
            totals.append(round(total, 2))
 
        df_trend = pd.DataFrame({"Month": months, "Total Cost ($)": totals})
        fig = px.line(df_trend, x="Month", y="Total Cost ($)", markers=True, title="📈 Monthly Cloud Cost Trend")
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Live AWS Cost Data Loaded Successfully.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch live AWS data: {e}")
        st.write("Falling back to demo trend data.")
        df_trend = pd.DataFrame({
            "Month": ["June", "July", "Aug", "Sep", "Oct"],
            "Total Cost ($)": [6000, 5900, 5400, 5700, 5200],
        })
        fig = px.line(df_trend, x="Month", y="Total Cost ($)", markers=True, title="📈 Monthly Cloud Cost Trend")
        st.plotly_chart(fig, use_container_width=True)
 
# -----------------------------
# AI Assistant Page
# -----------------------------
elif menu == "AI Assistant":
    st.header("💬 Agentic AI Assistant")
    st.markdown("Ask questions about your cloud usage, costs, or optimizations.")
 
    user_query = st.text_input("Ask something like: *Find idle EC2 instances or predict next month's cost...*")
 
    if st.button("Ask AI") and user_query:
        with st.spinner("🧠 Thinking..."):
            try:
                messages = [
                    SystemMessage(content="You are an AI assistant specialized in AWS cost optimization and cloud infrastructure analysis."),
                    HumanMessage(content=user_query)
                ]
                response = llm.invoke(messages)
                st.markdown(f"**🤖 AI:** {response.content}")
            except Exception as e:
                st.error(f"Error: {e}")
 
    st.markdown("""
    ---
    **Examples:**  
    • “Which service is contributing most to cost?”  
    • “How can I reduce idle resources?”  
    • “Show me cost-saving suggestions for EC2.”  
    """)
 
# -----------------------------
# About Page
# -----------------------------
elif menu == "About":
    st.header("📘 About the Project")
    st.write("""
    **Agentic AI for Smart Cloud Infrastructure Management**  
    Buildathon project focused on:
    - Automating cloud cost analysis using AI  
    - Integrating with AWS APIs (Cost Explorer, CloudWatch)  
    - Providing actionable insights  
    - Enabling chat-based cloud management  
    """)
    st.markdown("🧩 Built with: `LangChain`, `OpenAI GPT-4`, `Streamlit`, `Plotly`, and `AWS SDK (boto3)`")