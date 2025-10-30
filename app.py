# app.py — Agentic AI for AWS (Light ChatGPT Style UI + Extended Services)

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

def list_ec2():
    def _():
        res = clients["ec2"].describe_instances()
        return [
            i["InstanceId"]
            for r in res.get("Reservations", [])
            for i in r.get("Instances", [])
        ]
    return safe_call(_, [])

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

def list_rds():
    def _():
        res = clients["rds"].describe_db_instances()
        return [
            {
                "DBInstanceIdentifier": db["DBInstanceIdentifier"],
                "Engine": db["Engine"],
                "Status": db["DBInstanceStatus"],
                "Endpoint": db.get("Endpoint", {}).get("Address", "N/A")
            }
            for db in res.get("DBInstances", [])
        ]
    return safe_call(_, [])

def list_dynamodb():
    def _():
        res = clients["dynamodb"].list_tables()
        return res.get("TableNames", [])
    return safe_call(_, [])

def list_ecr():
    def _():
        res = clients["ecr"].describe_repositories()
        return [r["repositoryName"] for r in res.get("repositories", [])]
    return safe_call(_, [])

def list_iam_users():
    def _():
        res = clients["iam"].list_users()
        return [u["UserName"] for u in res.get("Users", [])]
    return safe_call(_, [])

# ---------------------- COST DATA ----------------------
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

def get_service_costs():
    ce = clients["ce"]
    today = datetime.utcnow().date()
    start = pd.Timestamp(today).replace(day=1).strftime("%Y-%m-%d")
    end = (today + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}]
    )
    services, costs = [], []
    for group in resp["ResultsByTime"][0]["Groups"]:
        services.append(group["Keys"][0])
        costs.append(float(group["Metrics"]["UnblendedCost"]["Amount"]))
    df = pd.DataFrame({"Service": services, "Cost": costs})
    df = df[df["Cost"] > 0].sort_values(by="Cost", ascending=False)
    return df

# ---------------------- SMART QUERY HANDLER ----------------------
def process_query(q: str):
    ql = q.lower()

    if "rds" in ql:
        rds = list_rds()
        if not rds:
            return "🚫 No active RDS instances found."
        st.markdown("### 🧩 Active RDS Instances")
        st.dataframe(pd.DataFrame(rds))
        return f"✅ Found {len(rds)} RDS instances."

    elif "dynamodb" in ql:
        tables = list_dynamodb()
        return f"🧮 DynamoDB Tables ({len(tables)}):\n" + ("\n".join(tables) if tables else "No DynamoDB tables found.")

    elif "ecr" in ql or "container" in ql or "repository" in ql:
        repos = list_ecr()
        return f"📦 ECR Repositories ({len(repos)}):\n" + ("\n".join(repos) if repos else "No ECR repositories found.")

    elif "iam" in ql or "user" in ql:
        users = list_iam_users()
        return f"👤 IAM Users ({len(users)}):\n" + ("\n".join(users) if users else "No IAM users found.")

    elif "ec2" in ql:
        ec2s = list_ec2()
        return f"🖥️ EC2 Instances ({len(ec2s)}):\n" + ("\n".join(ec2s) if ec2s else "No EC2 instances running.")

    elif "s3" in ql:
        s3s = list_s3()
        return f"🪣 S3 Buckets ({len(s3s)}):\n" + ("\n".join(s3s) if s3s else "No S3 buckets found.")

    elif "lambda" in ql:
        l = list_lambda()
        return f"⚙️ Lambda Functions ({len(l)}):\n" + ("\n".join(l) if l else "No Lambda functions deployed.")

    elif "cost" in ql or "bill" in ql or "spend" in ql:
        df = get_monthly_costs()
        df_service = get_service_costs()

        if len(df) >= 2:
            current, previous = df["Cost"].iloc[-1], df["Cost"].iloc[-2]
            diff = current - previous
            trend = "📈 increased" if diff > 0 else "📉 decreased"
            st.markdown(f"### 💰 AWS Cost Summary")
            st.markdown(f"**Current Month:** ${current:.2f}  \n**Previous Month:** ${previous:.2f}  \n**Change:** {trend} by ${abs(diff):.2f}")

        st.markdown("### 🧩 Service-wise Cost Breakdown")
        st.plotly_chart(px.bar(df_service, x="Service", y="Cost", title="AWS Service Cost Breakdown"), use_container_width=True)
        st.markdown("### 📊 Monthly Cost Trend")
        st.plotly_chart(px.line(df, x="Month", y="Cost", markers=True, title="AWS Monthly Cost (Last 6 Months)"), use_container_width=True)
        return ""

    elif any(word in ql for word in ["list services", "using services", "active services", "currently using", "which services"]):
        ec2s, s3s, lambdas, rds, dynamo, ecr, iam = list_ec2(), list_s3(), list_lambda(), list_rds(), list_dynamodb(), list_ecr(), list_iam_users()
        summary = [
            f"🖥️ EC2: {len(ec2s)}",
            f"🪣 S3: {len(s3s)}",
            f"⚙️ Lambda: {len(lambdas)}",
            f"🧩 RDS: {len(rds)}",
            f"🧮 DynamoDB: {len(dynamo)}",
            f"📦 ECR: {len(ecr)}",
            f"👤 IAM Users: {len(iam)}"
        ]
        return "### 🔍 Active AWS Services Summary\n" + "\n".join(summary)

    else:
        res = llm.invoke([
            SystemMessage(content="You are an AWS assistant with access to EC2, S3, Lambda, RDS, DynamoDB, ECR, IAM, and cost data."),
            HumanMessage(content=q)
        ])
        return res.content

# ---------------------- SIDEBAR NAV ----------------------
page = st.sidebar.radio("Navigation", ["🏠 Home", "💬 Chat Assistant", "📊 Cost Insights"])

# ---------------------- HOME ----------------------
if page == "🏠 Home":
    st.title("🏠 AWS Overview Dashboard")

    ec2s, s3s, lambdas = list_ec2(), list_s3(), list_lambda()
    df_cost = get_monthly_costs()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("EC2 Instances", len(ec2s))
        st.metric("S3 Buckets", len(s3s))
        st.metric("Lambda Functions", len(lambdas))
    with col2:
        if not df_cost.empty:
            st.metric("Current Month Cost ($)", round(df_cost["Cost"].iloc[-1], 2))
            if len(df_cost) > 1:
                st.metric("Previous Month Cost ($)", round(df_cost["Cost"].iloc[-2], 2))

    st.markdown("### 📊 Resource Usage Overview")
    df_usage = pd.DataFrame({
        "Service": ["EC2", "S3", "Lambda"],
        "Count": [len(ec2s), len(s3s), len(lambdas)]
    })

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df_usage, x="Service", y="Count", title="AWS Resources Count"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(df_usage, names="Service", values="Count", title="AWS Resource Distribution"), use_container_width=True)

    st.markdown("### 💵 Monthly Cost Trend")
    if not df_cost.empty:
        st.plotly_chart(px.line(df_cost, x="Month", y="Cost", markers=True, title="AWS Cost (Last 6 Months)"), use_container_width=True)
    else:
        st.info("No cost data available.")

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
