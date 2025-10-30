# app.py — Agentic AI for AWS (with color-coded EC2 state display)

import os
import pandas as pd
import streamlit as st
import boto3
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import plotly.express as px

# ---------------------- LOAD ENV ----------------------
load_dotenv()
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION") or "us-east-1"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("⚠️ Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Agentic AI • Smart AWS", layout="wide")

# ---------------------- LIGHT CHATGPT THEME ----------------------
st.markdown("""
<style>
body {
    background-color: #f9f9f9;
    color: #111;
    font-family: "Inter", sans-serif;
}
.stApp {
    background-color: #ffffff;
    color: #111;
}
h1, h2, h3, h4 {
    color: #111;
}
.stSidebar {
    background-color: #f1f3f6 !important;
    color: #111 !important;
}
[data-testid="stSidebarNav"] {
    background-color: #f1f3f6 !important;
}
.stChatMessage {
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.stChatMessage.user {
    background-color: #e8e8ea;
    text-align: right;
}
.stChatMessage.assistant {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- INIT ----------------------
@st.cache_resource
def init_llm():
    return ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o-mini", temperature=0.2)

@st.cache_resource
def init_aws_clients():
    session = boto3.Session(
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION
    )
    return {
        "session": session,
        "ec2": session.client("ec2"),
        "s3": session.client("s3"),
        "lambda": session.client("lambda"),
        "rds": session.client("rds"),
        "dynamodb": session.client("dynamodb"),
        "ecr": session.client("ecr"),
        "iam": session.client("iam"),
        "ce": session.client("ce", region_name="us-east-1")
    }

llm = init_llm()
clients = init_aws_clients()

# ---------------------- HELPERS ----------------------
def safe_call(fn, default=None):
    try:
        return fn()
    except Exception as e:
        return default if default is not None else f"⚠️ {e}"

def list_ec2_details():
    """Return detailed EC2 info (ID, Name, Type, State, LaunchTime)."""
    def _():
        res = clients["ec2"].describe_instances()
        data = []
        for r in res.get("Reservations", []):
            for i in r.get("Instances", []):
                name_tag = next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Name"), "N/A")
                data.append({
                    "Instance ID": i["InstanceId"],
                    "Name": name_tag,
                    "Type": i["InstanceType"],
                    "State": i["State"]["Name"],
                    "Launch Time": i["LaunchTime"].strftime("%Y-%m-%d %H:%M:%S")
                })
        df = pd.DataFrame(data)
        if not df.empty:
            df["State"] = df["State"].apply(color_state)
        return df
    return safe_call(_, pd.DataFrame())

def color_state(state):
    """Return HTML colored label for EC2 state."""
    colors = {
        "running": "#16a34a",   # green
        "stopped": "#dc2626",   # red
        "pending": "#facc15",   # yellow
        "shutting-down": "#f97316",
        "terminated": "#6b7280"
    }
    color = colors.get(state.lower(), "#9ca3af")
    return f'<b style="color:{color}">{state.capitalize()}</b>'

def list_s3():
    def _():
        res = clients["s3"].list_buckets()
        return [b["Name"] for b in res.get("Buckets", [])]
    return safe_call(_, [])

def list_lambda():
    def _():
        res = clients["lambda"].list_functions()
        return [f["FunctionName"] for f in res.get("Functions", [])]
    return safe_call(_, [])

def get_monthly_costs(months=6):
    ce = clients["ce"]
    today = datetime.utcnow().date()
    start = (pd.Timestamp(today).replace(day=1) - pd.DateOffset(months=months - 1)).date().strftime("%Y-%m-%d")
    end = (today + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"]
    )
    months, totals = [], []
    for r in resp["ResultsByTime"]:
        months.append(r["TimePeriod"]["Start"])
        totals.append(float(r["Total"]["UnblendedCost"]["Amount"]))
    df = pd.DataFrame({"Month": months, "Cost": totals})
    df["Month"] = pd.to_datetime(df["Month"]).dt.strftime("%b %Y")
    return df

# ---------------------- SMART QUERY HANDLER ----------------------
def process_query(q: str):
    ql = q.lower()

    # EC2 Smart Filtering
    if "ec2" in ql and any(w in ql for w in ["list", "show", "instances", "running", "stopped"]):
        df = list_ec2_details()
        if df.empty:
            return "No EC2 instances found."

        # Filter logic
        if "running" in ql:
            df = df[df["State"].str.contains("running", case=False, regex=True)]
            title = "🟢 Running EC2 Instances"
        elif "stopped" in ql:
            df = df[df["State"].str.contains("stopped", case=False, regex=True)]
            title = "🔴 Stopped EC2 Instances"
        else:
            title = "🖥️ All EC2 Instances"

        if df.empty:
            return f"No {title.lower()} found."

        st.markdown(f"### {title}")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        return f"✅ {title} displayed with color indicators."

    elif "s3" in ql and any(w in ql for w in ["list", "show", "buckets"]):
        s3s = list_s3()
        return f"🪣 **S3 Buckets ({len(s3s)})**:\n" + ("\n".join(s3s) if s3s else "No S3 buckets found.")

    elif any(word in ql for word in ["cost", "bill", "spend", "charges", "pricing"]):
        df = get_monthly_costs()
        if not df.empty:
            st.plotly_chart(px.line(df, x="Month", y="Cost", markers=True, title="AWS Monthly Cost"), use_container_width=True)
        return ""

    else:
        res = llm.invoke([
            SystemMessage(content="You are an AWS expert assistant that explains and helps manage AWS resources."),
            HumanMessage(content=q)
        ])
        return res.content


# ---------------------- SIDEBAR NAV ----------------------
page = st.sidebar.radio("Navigation", ["🏠 Home", "💬 Chat Assistant", "📊 Cost Insights"])

# ---------------------- HOME ----------------------
if page == "🏠 Home":
    st.title("🏠 AWS Overview Dashboard")

    df_ec2 = list_ec2_details()
    s3s, lambdas = list_s3(), list_lambda()
    df_cost = get_monthly_costs()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("EC2 Instances", len(df_ec2))
        st.metric("S3 Buckets", len(s3s))
        st.metric("Lambda Functions", len(lambdas))
    with col2:
        if not df_cost.empty:
            st.metric("Current Month Cost ($)", round(df_cost["Cost"].iloc[-1], 2))

    st.markdown("### 🖥️ EC2 Instance Status")
    if not df_ec2.empty:
        st.markdown(df_ec2.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No EC2 instances found.")

    st.markdown("### 💵 Monthly Cost Trend")
    if not df_cost.empty:
        st.plotly_chart(px.line(df_cost, x="Month", y="Cost", markers=True, title="AWS Cost (6 Months)"), use_container_width=True)

# ---------------------- CHAT ----------------------
elif page == "💬 Chat Assistant":
    st.title("💬 Agentic AI Chat for AWS")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about your AWS...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your AWS data..."):
                reply = process_query(user_input)
                if reply:
                    st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------- COST INSIGHTS ----------------------
elif page == "📊 Cost Insights":
    st.title("📊 AWS Cost Explorer")
    df_cost = get_monthly_costs()
    if df_cost.empty:
        st.warning("No cost data found.")
    else:
        st.plotly_chart(px.bar(df_cost, x="Month", y="Cost", title="Monthly AWS Cost (USD)"), use_container_width=True)
        st.plotly_chart(px.line(df_cost, x="Month", y="Cost", markers=True, title="Cost Trend Over Time"), use_container_width=True)

st.caption("⚡ Agentic AI • AWS Intelligence Dashboard")
