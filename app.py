# app.py — Agentic AI for AWS (Full Enhanced Dashboard)
import os
import pandas as pd
import streamlit as st
import boto3
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------- LOAD ENV ----------------------
load_dotenv()
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION") or "us-east-1"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("⚠️ Please set OPENAI_API_KEY in your .env file or repository secrets.")
    st.stop()

# ---------------------- PAGE CONFIG & THEME ----------------------
st.set_page_config(page_title="Agentic AI • Smart AWS", layout="wide", page_icon="⚡")

st.markdown("""
<style>
body {background-color:#fafafa;color:#111;font-family:Inter, sans-serif;}
.stApp {background-color:#ffffff;color:#111;}
.stSidebar {background-color:#f3f4f6 !important;color:#111 !important;}
.metric-card {background-color:#f9fafb;border-radius:12px;padding:14px;text-align:center;
box-shadow:0 2px 6px rgba(0,0,0,0.06);}
.dataframe th {background-color:#f1f1f1;padding:8px;}
.dataframe td {padding:6px;}
footer {visibility:hidden;}
.badge {padding:6px 8px;border-radius:6px;color:#fff;font-weight:600;}
.badge-green {background:#16a34a;}
.badge-red {background:#dc2626;}
.badge-yellow {background:#f59e0b;}
.badge-gray {background:#6b7280;}
</style>
""", unsafe_allow_html=True)

# ---------------------- CLIENTS (cached) ----------------------
@st.cache_resource
def init_llm():
    return ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o-mini", temperature=0.2)

@st.cache_resource
def init_aws_clients():
    # Use provided keys if available, otherwise rely on environment / role
    if AWS_KEY and AWS_SECRET:
        session = boto3.Session(
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            region_name=AWS_REGION
        )
    else:
        session = boto3.Session(region_name=AWS_REGION)
    # Cost Explorer must be queried in us-east-1
    ce_session = boto3.Session(
        aws_access_key_id=AWS_KEY or None,
        aws_secret_access_key=AWS_SECRET or None,
        region_name="us-east-1"
    )
    return {
        "session": session,
        "ec2": session.client("ec2"),
        "s3": session.client("s3"),
        "lambda": session.client("lambda"),
        "rds": session.client("rds"),
        "eks": session.client("eks"),
        "ecr": session.client("ecr"),
        "cloudfront": session.client("cloudfront"),
        "dynamodb": session.client("dynamodb"),
        "iam": session.client("iam"),
        "apigateway": session.client("apigateway"),
        "cloudwatch": session.client("cloudwatch"),
        "ce": ce_session.client("ce")
    }

llm = init_llm()
clients = init_aws_clients()

# ---------------------- SAFE CALL ----------------------
def safe_call(fn, default=None):
    try:
        return fn()
    except Exception as e:
        # return default or an informative string
        return default if default is not None else f"⚠️ {e}"

# ---------------------- UTILS ----------------------
def fmt_badge(state: str) -> str:
    s = (state or "").lower()
    if "running" in s or "available" in s or "active" in s:
        cls = "badge-green"
    elif "stopped" in s or "failed" in s or "inactive" in s:
        cls = "badge-red"
    elif "creating" in s or "pending" in s:
        cls = "badge-yellow"
    else:
        cls = "badge-gray"
    # HTML label
    return f'<span class="badge {cls}">{state}</span>'

# ---------------------- AWS LIST FUNCTIONS (cached) ----------------------
@st.cache_data(ttl=60)
def list_ec2_details():
    def _():
        res = clients["ec2"].describe_instances()
        rows = []
        for r in res.get("Reservations", []):
            for i in r.get("Instances", []):
                name = next((t["Value"] for t in i.get("Tags", []) if t.get("Key") == "Name"), "N/A")
                launch = i.get("LaunchTime")
                launch_str = launch.strftime("%Y-%m-%d %H:%M:%S") if launch else "-"
                rows.append({
                    "Instance ID": i.get("InstanceId"),
                    "Name": name,
                    "Type": i.get("InstanceType"),
                    "State": i.get("State", {}).get("Name"),
                    "Launch Time": launch_str
                })
        df = pd.DataFrame(rows)
        # add colored badge HTML
        if not df.empty:
            df["State_html"] = df["State"].apply(lambda s: fmt_badge(s))
        return df
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=60)
def list_s3_details():
    def _():
        res = clients["s3"].list_buckets()
        rows = []
        for b in res.get("Buckets", []):
            name = b.get("Name")
            created = b.get("CreationDate")
            created_str = created.strftime("%Y-%m-%d") if created else "-"
            rows.append({"Bucket": name, "Created": created_str})
        return pd.DataFrame(rows)
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=60)
def list_lambda_details():
    def _():
        res = clients["lambda"].list_functions()
        rows = []
        for f in res.get("Functions", []):
            rows.append({
                "Function Name": f.get("FunctionName"),
                "Runtime": f.get("Runtime", "-"),
                "Last Modified": f.get("LastModified", "-")
            })
        return pd.DataFrame(rows)
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=60)
def list_rds_details():
    def _():
        res = clients["rds"].describe_db_instances()
        rows = []
        for db in res.get("DBInstances", []):
            created = db.get("InstanceCreateTime")
            created_str = created.strftime("%Y-%m-%d") if created else "-"
            rows.append({
                "DB Identifier": db.get("DBInstanceIdentifier"),
                "Engine": db.get("Engine"),
                "Status": db.get("DBInstanceStatus"),
                "Class": db.get("DBInstanceClass"),
                "Endpoint": db.get("Endpoint", {}).get("Address", "-"),
                "Created": created_str
            })
        return pd.DataFrame(rows)
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=60)
def list_eks_clusters():
    def _():
        clusters = clients["eks"].list_clusters().get("clusters", [])
        rows = []
        for name in clusters:
            desc = clients["eks"].describe_cluster(name=name).get("cluster", {})
            created = desc.get("createdAt")
            created_str = created.strftime("%Y-%m-%d") if created else "-"
            rows.append({
                "Cluster Name": name,
                "Status": desc.get("status", "-"),
                "Version": desc.get("version", "-"),
                "Created": created_str
            })
        return pd.DataFrame(rows)
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=60)
def list_ecr_repos():
    def _():
        res = clients["ecr"].describe_repositories()
        return pd.DataFrame([{"Repository": r.get("repositoryName"), "URI": r.get("repositoryUri")} for r in res.get("repositories", [])])
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=60)
def list_cloudfront():
    def _():
        res = clients["cloudfront"].list_distributions()
        items = res.get("DistributionList", {}).get("Items", [])
        return pd.DataFrame([{"Id": d["Id"], "Domain": d["DomainName"], "Status": d["Status"], "Enabled": d["Enabled"]} for d in items])
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=60)
def list_dynamodb_tables():
    def _():
        res = clients["dynamodb"].list_tables()
        return pd.DataFrame([{"Table": t} for t in res.get("TableNames", [])])
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=60)
def list_iam_users():
    def _():
        res = clients["iam"].list_users()
        return pd.DataFrame([{"UserName": u.get("UserName"), "Created": u.get("CreateDate").strftime("%Y-%m-%d") if u.get("CreateDate") else "-"} for u in res.get("Users", [])])
    return safe_call(_, pd.DataFrame())

# ---------------------- COSTS ----------------------
@st.cache_data(ttl=300)
def get_monthly_costs(months=6):
    def _():
        ce = clients["ce"]
        today = datetime.utcnow().date()
        start_dt = (pd.Timestamp(today).replace(day=1) - pd.DateOffset(months=months - 1)).date()
        start = start_dt.strftime("%Y-%m-%d")
        end = (today + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        resp = ce.get_cost_and_usage(TimePeriod={"Start": start, "End": end}, Granularity="MONTHLY", Metrics=["UnblendedCost"])
        months_list = []
        totals = []
        for r in resp.get("ResultsByTime", []):
            months_list.append(r["TimePeriod"]["Start"])
            totals.append(float(r["Total"]["UnblendedCost"]["Amount"]))
        df = pd.DataFrame({"Month": months_list, "Cost": totals})
        if not df.empty:
            df["Month"] = pd.to_datetime(df["Month"]).dt.strftime("%b %Y")
        return df
    return safe_call(_, pd.DataFrame())

@st.cache_data(ttl=300)
def get_service_costs_for_month(target_date=None):
    """
    Return cost per AWS SERVICE for the month that contains target_date (defaults to current month).
    """
    def _():
        ce = clients["ce"]
        if target_date is None:
            target_date = datetime.utcnow().date()
        # first day of target month
        start = pd.Timestamp(target_date).replace(day=1).strftime("%Y-%m-%d")
        # first day of next month
        next_month = (pd.Timestamp(start) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": next_month},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}]
        )
        results = resp.get("ResultsByTime", [])
        if not results:
            return pd.DataFrame()
        groups = results[0].get("Groups", [])
        rows = []
        for g in groups:
            svc = g.get("Keys", [None])[0]
            amount = float(g.get("Metrics", {}).get("UnblendedCost", {}).get("Amount", 0.0))
            rows.append({"Service": svc or "Unknown", "Cost": amount})
        df = pd.DataFrame(rows).sort_values("Cost", ascending=False)
        return df
    return safe_call(_, pd.DataFrame())

# ---------------------- NEW: ESTIMATOR ----------------------
def estimate_service_costs(total_cost: float | None = None):
    """
    Estimate service costs using simple per-resource unit rates and counts.
    If total_cost provided and >0, scale the totals to match it.
    """
    # gather counts
    ec2 = list_ec2_details()
    s3 = list_s3_details()
    rds = list_rds_details()
    eks = list_eks_clusters()
    lamb = list_lambda_details()
    ecr = list_ecr_repos()
    cf = list_cloudfront()
    ddb = list_dynamodb_tables()
    iam = list_iam_users()

    counts = {
        "EC2": len(ec2),
        "S3": len(s3),
        "RDS": len(rds),
        "EKS": len(eks),
        "Lambda": len(lamb),
        "ECR": len(ecr),
        "CloudFront": len(cf),
        "DynamoDB": len(ddb),
        "IAM Users": len(iam)
    }

    # basic per-unit monthly rates (very rough heuristics for demo)
    rates = {
        "EC2": 30.0,        # $/instance-month
        "S3": 1.0,          # $/bucket-month (placeholder; actual depends on bytes stored)
        "RDS": 80.0,        # $/db-month
        "EKS": 40.0,        # $/cluster-month
        "Lambda": 3.0,      # $/function-month (placeholder)
        "ECR": 2.0,         # $/repo-month
        "CloudFront": 5.0,  # $/distribution-month (placeholder)
        "DynamoDB": 10.0,   # $/table-month (placeholder)
        "IAM Users": 0.2    # $/user-month (very small)
    }

    rows = []
    for svc, cnt in counts.items():
        est = rates.get(svc, 1.0) * max(cnt, 0)
        rows.append({"Service": svc, "Count": cnt, "EstimatedCost": est})
    est_df = pd.DataFrame(rows).sort_values("EstimatedCost", ascending=False).reset_index(drop=True)

    # scale to match total_cost if provided and > 0
    if total_cost and total_cost > 0 and not est_df.empty:
        total_est = est_df["EstimatedCost"].sum()
        if total_est > 0:
            est_df["EstimatedCostScaled"] = est_df["EstimatedCost"] * (total_cost / total_est)
        else:
            est_df["EstimatedCostScaled"] = est_df["EstimatedCost"]
    else:
        est_df["EstimatedCostScaled"] = est_df["EstimatedCost"]

    return est_df

# ---------------------- SMART QUERY PROCESSOR ----------------------
def process_query(q: str):
    ql = q.lower().strip()
    # quick mapping
    if any(k in ql for k in ["which services", "active services", "currently using", "what services"]):
        # gather counts and creation/launch info + last month cost per service
        ec2 = list_ec2_details()
        s3 = list_s3_details()
        rds = list_rds_details()
        eks = list_eks_clusters()
        lamb = list_lambda_details()
        ecr = list_ecr_repos()
        cf = list_cloudfront()
        ddb = list_dynamodb_tables()
        iam = list_iam_users()

        svc_rows = []
        # last-month service costs
        svc_costs = get_service_costs_for_month()
        cost_map = dict(zip(svc_costs.get("Service", []), svc_costs.get("Cost", [])))

        def svc_cost(svc_name):
            return cost_map.get(svc_name, 0.0)

        # EC2
        svc_rows.append({
            "Service": "EC2",
            "Count": len(ec2),
            "Example resource / created": ec2["Launch Time"].iloc[0] if not ec2.empty else "-",
            "Last month cost ($)": round(svc_cost("Amazon Elastic Compute Cloud - Compute"), 2) if svc_cost("Amazon Elastic Compute Cloud - Compute") else round(svc_cost("Amazon Elastic Compute Cloud"), 2)
        })
        # S3
        svc_rows.append({
            "Service": "S3",
            "Count": len(s3),
            "Example resource / created": s3["Created"].iloc[0] if not s3.empty else "-",
            "Last month cost ($)": round(svc_cost("Amazon Simple Storage Service"), 2)
        })
        # RDS
        svc_rows.append({
            "Service": "RDS",
            "Count": len(rds),
            "Example resource / created": rds["Created"].iloc[0] if not rds.empty else "-",
            "Last month cost ($)": round(svc_cost("Amazon Relational Database Service"), 2)
        })
        # EKS
        svc_rows.append({
            "Service": "EKS",
            "Count": len(eks),
            "Example resource / created": eks["Created"].iloc[0] if not eks.empty else "-",
            "Last month cost ($)": round(svc_cost("Amazon Elastic Kubernetes Service"), 2)
        })
        # Lambda
        svc_rows.append({
            "Service": "Lambda",
            "Count": len(lamb),
            "Example resource / created": lamb["Last Modified"].iloc[0] if not lamb.empty else "-",
            "Last month cost ($)": round(svc_cost("AWS Lambda"), 2)
        })
        # ECR
        svc_rows.append({
            "Service": "ECR",
            "Count": len(ecr),
            "Example resource / created": ecr["Repository"].iloc[0] if not ecr.empty else "-",
            "Last month cost ($)": round(svc_cost("Amazon Elastic Container Registry (ECR)"), 2)
        })
        # CloudFront
        svc_rows.append({
            "Service": "CloudFront",
            "Count": len(cf),
            "Example resource / created": cf["Domain"].iloc[0] if not cf.empty else "-",
            "Last month cost ($)": round(svc_cost("Amazon CloudFront"), 2)
        })
        # DynamoDB
        svc_rows.append({
            "Service": "DynamoDB",
            "Count": len(ddb),
            "Example resource / created": ddb["Table"].iloc[0] if not ddb.empty else "-",
            "Last month cost ($)": round(svc_cost("Amazon DynamoDB"), 2)
        })
        # IAM
        svc_rows.append({
            "Service": "IAM Users",
            "Count": len(iam),
            "Example resource / created": iam["Created"].iloc[0] if not iam.empty else "-",
            "Last month cost ($)": 0.0
        })

        svc_df = pd.DataFrame(svc_rows)
        st.markdown("### 🔍 Active Services Summary")
        st.dataframe(svc_df, use_container_width=True)

        # show bar / pie of costs for only services with cost > 0
        cost_df = get_service_costs_for_month()
        # --- NEW: estimated costs scaled to last-month total if we have monthly total ---
        monthly = get_monthly_costs()
        last_month_total = monthly["Cost"].iloc[-1] if (not monthly.empty) else None
        est_df = estimate_service_costs(total_cost=last_month_total)

        if not cost_df.empty:
            st.markdown("### 🧾 Last Month Cost by Service (Cost Explorer)")
            st.plotly_chart(px.bar(cost_df, x="Service", y="Cost", title="Service cost (last month)", text_auto=True), use_container_width=True)
            st.plotly_chart(px.pie(cost_df.head(10), names="Service", values="Cost", title="Top services by cost (last month)"), use_container_width=True)
        else:
            st.info("No service cost data available from Cost Explorer for the last month (Cost Explorer may be off / permission missing).")

        # show estimated breakdown and compare
        st.markdown("### 🔮 Estimated Service Cost (heuristic) — scaled to total if available")
        est_display = est_df[["Service", "Count", "EstimatedCost", "EstimatedCostScaled"]].rename(columns={"EstimatedCostScaled":"EstimatedScaled"})
        st.dataframe(est_display, use_container_width=True)

        # comparison table
        if not cost_df.empty:
            # prepare comparison
            merged = pd.merge(cost_df, est_df[["Service", "EstimatedCostScaled"]], on="Service", how="outer").fillna(0)
            merged = merged.rename(columns={"Cost":"CE_Cost", "EstimatedCostScaled":"Est_Cost"})
            merged["Diff"] = merged["CE_Cost"] - merged["Est_Cost"]
            merged = merged.sort_values("CE_Cost", ascending=False).reset_index(drop=True)
            st.markdown("### ⚖️ CE vs Estimated (last month)")
            st.dataframe(merged.head(30), use_container_width=True)

        # AI quick insights (small)
        insight_lines = []
        if not cost_df.empty:
            top = cost_df.head(3)
            insight_lines.append(f"Top services by CE cost: {', '.join(top['Service'].tolist())}.")
        if last_month_total:
            insight_lines.append(f"Total billed last month: ${last_month_total:.2f}.")
        # some simple heuristic suggestions
        if len(ec2) > 0:
            idle_ec2 = ec2[ec2["State"].str.contains("stopped", case=False, na=False)]
            if not idle_ec2.empty:
                insight_lines.append(f"Found {len(idle_ec2)} stopped EC2 instances — consider terminating or scheduling them.")
        if len(rds) > 0:
            insight_lines.append("Check RDS backups and right-sizing for RDS instances to optimize costs.")
        if not cost_df.empty and est_df["EstimatedCostScaled"].sum() > 0:
            insight_lines.append("Compare CE and estimates to find tracking gaps (e.g., third-party charges, data transfer).")

        if insight_lines:
            st.markdown("#### 🧠 Quick Insights & Suggestions")
            for l in insight_lines:
                st.markdown(f"- {l}")

        return "✅ Active services summary displayed above."

    # EC2 specific queries
    if "ec2" in ql:
        df = list_ec2_details()
        if df.empty:
            return "No EC2 instances found."
        if "running" in ql:
            filtered = df[df["State"].str.contains("running", case=False, na=False)]
            st.markdown("### 🟢 Running EC2 Instances")
            st.markdown(filtered.to_html(escape=False, index=False), unsafe_allow_html=True)
            return "✅ Running EC2 instances listed."
        elif "stopped" in ql:
            filtered = df[df["State"].str.contains("stopped", case=False, na=False)]
            st.markdown("### 🔴 Stopped EC2 Instances")
            st.markdown(filtered.to_html(escape=False, index=False), unsafe_allow_html=True)
            return "✅ Stopped EC2 instances listed."
        else:
            st.markdown("### 🖥️ All EC2 Instances")
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
            return "✅ EC2 instances listed."

    # RDS
    if "rds" in ql:
        df = list_rds_details()
        if df.empty:
            return "No RDS instances found."
        st.markdown("### 🗄️ RDS Instances")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        return "✅ RDS instances listed."

    # EKS
    if "eks" in ql:
        df = list_eks_clusters()
        if df.empty:
            return "No EKS clusters found."
        st.markdown("### ☸️ EKS Clusters")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        return "✅ EKS clusters listed."

    # Lambda
    if "lambda" in ql:
        df = list_lambda_details()
        if df.empty:
            return "No Lambda functions found."
        st.markdown("### ⚡ Lambda Functions")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        return "✅ Lambda functions listed."

    # S3
    if "s3" in ql:
        df = list_s3_details()
        if df.empty:
            return "No S3 buckets found."
        st.markdown("### 🪣 S3 Buckets")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        return "✅ S3 buckets listed."

    # ECR
    if "ecr" in ql:
        df = list_ecr_repos()
        if df.empty:
            return "No ECR repositories found."
        st.markdown("### 🐳 ECR Repositories")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        return "✅ ECR repositories listed."

    # CloudFront
    if "cloudfront" in ql:
        df = list_cloudfront()
        if df.empty:
            return "No CloudFront distributions found."
        st.markdown("### 🌍 CloudFront Distributions")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        return "✅ CloudFront distributions listed."

    # DynamoDB
    if "dynamodb" in ql or "dynamo" in ql:
        df = list_dynamodb_tables()
        if df.empty:
            return "No DynamoDB tables found."
        st.markdown("### 📚 DynamoDB Tables")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        return "✅ DynamoDB tables listed."

    # CloudWatch metrics (sample)
    if "cloudwatch" in ql:
        df = safe_call(lambda: clients["cloudwatch"].list_metrics(Namespace="AWS/EC2").get("Metrics", []), [])
        if not df:
            return "No CloudWatch metrics available or permission missing."
        st.markdown("### 📈 CloudWatch (sample EC2 metrics)")
        st.dataframe(pd.DataFrame(df[:12]))
        return "✅ CloudWatch sample metrics shown."

    # Cost explorer general
    if any(k in ql for k in ["cost", "spend", "bill", "charges"]):
        # if user asks specific "this month" or "current month", show numeric
        df_monthly = get_monthly_costs()
        svc_costs = get_service_costs_for_month()
        # numeric quick reply for "this month"
        if "this month" in ql or "current month" in ql or "this month's" in ql:
            if not df_monthly.empty:
                current = df_monthly["Cost"].iloc[-1]
                st.markdown(f"### 💰 Current month (approx): **${current:.2f}**")
            else:
                st.info("No monthly cost data available (Cost Explorer may be off).")

            # show breakdowns (both CE and estimated)
            if not svc_costs.empty:
                st.markdown("#### 🧾 Cost Explorer: service breakdown (current month)")
                st.plotly_chart(px.bar(svc_costs.head(12), x="Service", y="Cost", title="Service costs (current month)", text_auto=True), use_container_width=True)
            else:
                st.info("No service breakdown available via Cost Explorer.")

            # show estimated breakdown scaled to current month total if available
            total_now = df_monthly["Cost"].iloc[-1] if (not df_monthly.empty) else None
            est_now = estimate_service_costs(total_cost=total_now)
            st.markdown("#### 🔮 Estimated costs (heuristic)")
            st.dataframe(est_now[["Service", "Count", "EstimatedCostScaled"]].rename(columns={"EstimatedCostScaled":"Estimated($)"}), use_container_width=True)
            return "✅ Shown current month cost & breakdowns."

        # default cost view
        if not df_monthly.empty:
            st.markdown("### 💵 Monthly Cost Trend")
            st.plotly_chart(px.line(df_monthly, x="Month", y="Cost", markers=True, title="AWS Monthly Cost (last months)"), use_container_width=True)
        else:
            st.info("No monthly cost data available (Cost Explorer may be off).")
        if not svc_costs.empty:
            st.markdown("### 🧾 Service breakdown (this month)")
            st.plotly_chart(px.bar(svc_costs.head(12), x="Service", y="Cost", title="Service costs (this month)", text_auto=True), use_container_width=True)
        return "✅ Cost information displayed."

    # Fallback to LLM for "how-to" and general questions (LLM uses OpenAI)
    try:
        resp = llm.invoke([SystemMessage(content="You are an AWS assistant that gives helpful answers and step-by-step instructions."), HumanMessage(content=q)])
        content = getattr(resp, "content", None) or str(resp)
        return content
    except Exception as e:
        return f"⚠️ LLM error: {e}"

# ---------------------- SIDEBAR & NAV ----------------------
page = st.sidebar.radio("Navigation", ["🏠 Home", "💬 Chat Assistant", "📊 Cost Insights"])

# ---------------------- HOME ----------------------
if page == "🏠 Home":
    st.title("🏠 AWS Overview Dashboard")
    # fetch summary data
    df_ec2 = list_ec2_details()
    df_s3 = list_s3_details()
    df_lambda = list_lambda_details()
    df_rds = list_rds_details()
    df_eks = list_eks_clusters()
    df_ecr = list_ecr_repos()
    df_cf = list_cloudfront()
    df_ddb = list_dynamodb_tables()
    df_iam = list_iam_users()
    df_cost = get_monthly_costs()

    # top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🖥️ EC2 Instances", len(df_ec2))
    c2.metric("🪣 S3 Buckets", len(df_s3))
    c3.metric("⚡ Lambda Functions", len(df_lambda))
    if not df_cost.empty:
        c4.metric("💰 This month (approx)", f"${df_cost['Cost'].iloc[-1]:.2f}")

    st.markdown("### 🔎 Service Counts (detailed)")
    counts = {
        "EC2": len(df_ec2), "S3": len(df_s3), "Lambda": len(df_lambda),
        "RDS": len(df_rds), "EKS": len(df_eks), "ECR": len(df_ecr),
        "CloudFront": len(df_cf), "DynamoDB": len(df_ddb), "IAM Users": len(df_iam)
    }
    counts_df = pd.DataFrame([counts]).T.reset_index()
    counts_df.columns = ["Service", "Count"]
    st.dataframe(counts_df, use_container_width=True)

    # show EC2 table with colored states
    st.markdown("### 🖥️ EC2 Instance Status")
    if df_ec2.empty:
        st.info("No EC2 instances found.")
    else:
        # show State badge column
        df_show = df_ec2.copy()
        df_show["State"] = df_show["State_html"]
        df_show = df_show.drop(columns=["State_html"])
        st.markdown(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)

    # small tables for other services
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🗄️ RDS")
        if df_rds.empty:
            st.write("None")
        else:
            st.markdown(df_rds.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("#### ⚡ Lambda")
        st.dataframe(df_lambda.head(10), use_container_width=True)
        st.markdown("#### 🐳 ECR")
        st.dataframe(df_ecr.head(10), use_container_width=True)
    with right:
        st.markdown("#### 🪣 S3")
        st.dataframe(df_s3.head(10), use_container_width=True)
        st.markdown("#### ☸️ EKS")
        st.dataframe(df_eks.head(10), use_container_width=True)
        st.markdown("#### 🌍 CloudFront")
        st.dataframe(df_cf.head(10), use_container_width=True)

    # costs
    st.markdown("### 💵 Cost Visualizations")
    if not df_cost.empty:
        st.plotly_chart(px.line(df_cost, x="Month", y="Cost", markers=True, title="AWS Monthly Cost (6 months)"), use_container_width=True)

    # --- ADDED: show both CE service costs and heuristic estimates side-by-side ---
    svc_costs = get_service_costs_for_month()
    # estimated (scaled to current month if available)
    total_current = df_cost["Cost"].iloc[-1] if (not df_cost.empty) else None
    est = estimate_service_costs(total_cost=total_current)

    st.markdown("### 🔀 Service Cost — Actual (Cost Explorer) vs Estimated (heuristic)")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### 🧾 Cost Explorer (this month)")
        if not svc_costs.empty:
            st.plotly_chart(px.bar(svc_costs.head(12), x="Service", y="Cost", title="CE: Service cost (current month)", text_auto=True), use_container_width=True)
            st.plotly_chart(px.pie(svc_costs.head(8), names="Service", values="Cost", title="CE: Top services (current month)"), use_container_width=True)
        else:
            st.info("No Cost Explorer service data available (permission or CE not enabled).")

    with cols[1]:
        st.markdown("#### 🔮 Estimated breakdown (heuristic)")
        if not est.empty:
            st.plotly_chart(px.bar(est, x="Service", y="EstimatedCostScaled", title="Estimated service cost (scaled)", text_auto=True), use_container_width=True)
            st.plotly_chart(px.pie(est.head(8), names="Service", values="EstimatedCostScaled", title="Estimated: Top services"), use_container_width=True)
            st.dataframe(est[["Service", "Count", "EstimatedCost", "EstimatedCostScaled"]].rename(columns={"EstimatedCostScaled":"EstimatedScaled($)"}), use_container_width=True)
        else:
            st.info("Estimates not available.")

    # comparison (if CE data present)
    if not svc_costs.empty and not est.empty:
        merged = pd.merge(svc_costs, est[["Service", "EstimatedCostScaled"]], on="Service", how="outer").fillna(0)
        merged = merged.rename(columns={"Cost":"CE_Cost", "EstimatedCostScaled":"Est_Cost"})
        merged["Diff"] = merged["CE_Cost"] - merged["Est_Cost"]
        st.markdown("#### ⚖️ CE vs Estimate (this month)")
        st.dataframe(merged.sort_values("CE_Cost", ascending=False).head(30), use_container_width=True)
    else:
        st.info("Comparison table not available (missing CE or estimate data).")

# ---------------------- CHAT ASSISTANT ----------------------
elif page == "💬 Chat Assistant":
    st.title("💬 Agentic AI Chat for AWS")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about your AWS environment (e.g., 'list running ec2', 'do I have RDS', 'show cost for last month')...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                reply = process_query(user_input)
                # reply can be text (string) or already-rendered charts/tables (we still echo a confirmation)
                if isinstance(reply, str) and reply:
                    st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply if isinstance(reply, str) else "Displayed results."})

# ---------------------- COST INSIGHTS ----------------------
elif page == "📊 Cost Insights":
    st.title("📊 AWS Cost Explorer")
    df_cost = get_monthly_costs()
    if df_cost.empty:
        st.warning("No Cost Explorer data available (enable Cost Explorer / check permissions).")
    else:
        st.plotly_chart(px.bar(df_cost, x="Month", y="Cost", title="Monthly AWS Cost (USD)", text_auto=True), use_container_width=True)
        st.plotly_chart(px.line(df_cost, x="Month", y="Cost", markers=True, title="Cost Trend Over Time"), use_container_width=True)
    svc_costs = get_service_costs_for_month()
    if not svc_costs.empty:
        st.markdown("### Service-wise cost (current month)")
        st.plotly_chart(px.bar(svc_costs.head(20), x="Service", y="Cost", title="Service cost (current month)", text_auto=True), use_container_width=True)

st.caption("⚡ FinMinds Agentic AI • AWS Intelligence Dashboard")
