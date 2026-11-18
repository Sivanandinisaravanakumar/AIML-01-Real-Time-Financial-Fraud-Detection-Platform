from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from kafka import KafkaConsumer

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "ieee_cis" / "processed"
MODEL_DIR = ROOT / "models" / "ieee_cis"
REPORTS_DIR = ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train_features.csv"
CASE_FILE = REPORTS_DIR / "analyst_cases.csv"
DRIFT_FILE = REPORTS_DIR / "retraining_trigger.json"
METRICS_FILE = MODEL_DIR / "metrics.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Fraud Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Financial Fraud Detection")
st.caption("IEEE-CIS | Fraud Analyst Monitoring Dashboard")

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_json(path):
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def create_case_file():
    if not TRAIN_FILE.exists():
        st.error(f"Training feature file not found: {TRAIN_FILE}")
        st.stop()

    required_columns = [
        "TransactionID",
        "TransactionAmt",
        "isFraud",
    ]

    df = pd.read_csv(
        TRAIN_FILE,
        usecols=required_columns,
    )

    fraud_cases = df[df["isFraud"] == 1].head(100).copy()

    fraud_cases["case_status"] = "Open"
    fraud_cases["analyst_feedback"] = "Pending"

    fraud_cases.to_csv(CASE_FILE, index=False)

    return fraud_cases


def load_cases():
    if not CASE_FILE.exists():
        return create_case_file()

    cases = pd.read_csv(CASE_FILE)

    if "case_status" not in cases.columns:
        cases["case_status"] = "Open"

    if "analyst_feedback" not in cases.columns:
        cases["analyst_feedback"] = "Pending"

    cases["case_status"] = cases["case_status"].fillna("Open")
    cases["analyst_feedback"] = cases["analyst_feedback"].fillna("Pending")

    return cases


def save_cases(edited_cases):
    temp_file = CASE_FILE.with_suffix(".tmp")
    edited_cases.to_csv(temp_file, index=False)
    temp_file.replace(CASE_FILE)


# --------------------------------------------------
# Kafka Consumer
# --------------------------------------------------

@st.cache_resource
def get_dashboard_consumer():
    return KafkaConsumer(
        "fraud-transactions",
        bootstrap_servers=["localhost:9092"],
        group_id="streamlit-live-alerts",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
        request_timeout_ms=10000,
    )


def read_live_alerts():
    consumer = get_dashboard_consumer()

    if "live_alerts" not in st.session_state:
        st.session_state.live_alerts = []

    messages = consumer.poll(
        timeout_ms=500,
        max_records=50,
    )

    for records in messages.values():
        for message in records:
            transaction = message.value

            st.session_state.live_alerts.append({
                "Received At": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Transaction ID": transaction.get("transaction_id"),
                "Fraud Probability": transaction.get(
                    "fraud_probability"
                ),
                "Threshold": transaction.get("threshold"),
                "Prediction": transaction.get("prediction"),
                "Label": transaction.get("label"),
            })

    st.session_state.live_alerts = (
        st.session_state.live_alerts[-500:]
    )


# --------------------------------------------------
# Load monitoring information
# --------------------------------------------------

drift = load_json(DRIFT_FILE)
model_metrics = load_json(METRICS_FILE)
cases = load_cases()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Dashboard Menu")
st.sidebar.write("Dataset: IEEE-CIS")

page = st.sidebar.radio(
    "Select View",
    [
        "Overview",
        "Analyst Case Queue",
        "Data Drift",
        "Live Kafka Alerts",
        "Fraud Network",
    ],
)

# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":

    st.header("Monitoring Overview")

    drift_count = drift.get("drifted_features", 0)
    total_features = drift.get("total_features", 0)
    drift_share = drift.get("drift_share", 0)

    retraining_required = drift.get(
        "retraining_required",
        False,
    )

    open_count = int(
        (cases["case_status"] == "Open").sum()
    )

    closed_count = int(
        (cases["case_status"] == "Closed").sum()
    )

    pr_auc = model_metrics.get("PR_AUC")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Drifted Features",
        f"{drift_count}/{total_features}",
    )

    col2.metric(
        "Drift Share",
        f"{drift_share * 100:.1f}%",
    )

    col3.metric(
        "Open Cases",
        open_count,
    )

    col4.metric(
        "XGBoost PR-AUC",
        f"{pr_auc:.4f}" if isinstance(pr_auc, (int, float)) else "N/A",
    )

    st.divider()

    st.subheader("System Status")

    if retraining_required:
        st.error(
            "Drift threshold reached. "
            "Model retraining review is required."
        )
    else:
        st.success(
            "Drift is below the configured threshold. "
            "No retraining request."
        )

    st.info(
        f"Closed cases: {closed_count} | "
        f"Total analyst cases: {len(cases)}"
    )

    st.warning(
        "The current case queue contains historical labelled "
        "IEEE-CIS fraud examples. It is not a live Kafka alert feed."
    )


# --------------------------------------------------
# Analyst Case Queue
# --------------------------------------------------

elif page == "Analyst Case Queue":

    st.header("Analyst Case Queue")

    st.write(
        "Review historical fraud examples and update their case status."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Cases", len(cases))

    col2.metric(
        "Open Cases",
        int((cases["case_status"] == "Open").sum()),
    )

    col3.metric(
        "Closed Cases",
        int((cases["case_status"] == "Closed").sum()),
    )

    st.divider()

    with st.form("analyst_case_form"):

        edited_cases = st.data_editor(
            cases,
            hide_index=True,
            use_container_width=True,
            key="analyst_cases_editor",
            disabled=[
                "TransactionID",
                "TransactionAmt",
                "isFraud",
            ],
            column_config={
                "TransactionID": st.column_config.NumberColumn(
                    "Transaction ID",
                ),
                "TransactionAmt": st.column_config.NumberColumn(
                    "Transaction Amount",
                    format="%.2f",
                ),
                "isFraud": st.column_config.NumberColumn(
                    "Actual Fraud Label",
                ),
                "case_status": st.column_config.SelectboxColumn(
                    "Case Status",
                    options=[
                        "Open",
                        "Investigating",
                        "Closed",
                    ],
                    required=True,
                ),
                "analyst_feedback": st.column_config.SelectboxColumn(
                    "Analyst Feedback",
                    options=[
                        "Pending",
                        "Confirmed Fraud",
                        "False Positive",
                        "Needs Review",
                    ],
                    required=True,
                ),
            },
        )

        save_clicked = st.form_submit_button(
            "Save Analyst Updates",
            type="primary",
            use_container_width=True,
        )

    if save_clicked:
        try:
            save_cases(edited_cases)

            st.success(
                "Analyst updates saved successfully!"
            )

            st.rerun()

        except (OSError, PermissionError, ValueError) as error:
            st.error(
                f"Could not save analyst updates: {error}"
            )

    st.caption(
        "Updates are saved locally in reports/analyst_cases.csv."
    )


# --------------------------------------------------
# Data Drift
# --------------------------------------------------

elif page == "Data Drift":

    st.header("Evidently Data Drift Monitoring")

    if not drift:
        st.warning(
            "Drift report data not found. "
            "Run src/drift_monitor.py first."
        )
    else:
        drift_count = drift.get("drifted_features", 0)
        total_features = drift.get("total_features", 0)
        drift_share = drift.get("drift_share", 0)
        threshold = drift.get("drift_threshold", 0.5)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Features", total_features)
        col2.metric("Drifted Features", drift_count)

        col3.metric(
            "Drift Share",
            f"{drift_share * 100:.1f}%",
        )

        st.progress(
            min(max(drift_share, 0.0), 1.0),
            text="Share of drifted features",
        )

        st.write(
            f"Configured retraining threshold: "
            f"{threshold * 100:.1f}%"
        )

        if drift.get("retraining_required", False):
            st.error(
                "Retraining review required: drift threshold reached."
            )
        else:
            st.success(
                "No retraining request: drift is below threshold."
            )

        st.json(drift)

    report_file = REPORTS_DIR / "ieee_data_drift_report.html"

    if report_file.exists():
        st.caption(
            "Detailed Evidently HTML report: "
            "reports/ieee_data_drift_report.html"
        )


# --------------------------------------------------
# Live Kafka Alerts
# --------------------------------------------------

elif page == "Live Kafka Alerts":

    st.header("Live Kafka Fraud Alerts")

    st.caption(
        "Real-time transactions from Kafka topic: "
        "fraud-transactions"
    )

    @st.fragment(run_every="3s")
    def display_live_alerts():

        try:
            read_live_alerts()

            alerts = st.session_state.live_alerts

            fraud_count = sum(
                1 for alert in alerts
                if alert["Prediction"] == 1
            )

            normal_count = len(alerts) - fraud_count

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Alerts", len(alerts))
            col2.metric("Fraud Alerts", fraud_count)
            col3.metric("Normal Transactions", normal_count)

            st.divider()

            if alerts:
                alert_df = pd.DataFrame(alerts)

                st.dataframe(
                    alert_df.iloc[::-1],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Waiting for new Kafka transactions..."
                )

        except Exception as error:
            st.error(
                f"Kafka connection or reading error: {error}"
            )

    display_live_alerts()



# --------------------------------------------------
# Fraud Network
# --------------------------------------------------

elif page == "Fraud Network":

    st.header("Shared Device Fraud Network")

    st.caption(
        "Card-device relationships from IEEE-CIS training data. "
        "Only devices shared by 2?20 cards are considered."
    )

    required_columns = [
        "TransactionID", "isFraud",
        "card1", "card2", "card3", "card5", "card6",
        "DeviceInfo",
    ]

    if not TRAIN_FILE.exists():
        st.error(f"Training data not found: {TRAIN_FILE}")
        st.stop()

    with st.spinner("Building card-device network..."):
        df = pd.read_csv(TRAIN_FILE, usecols=required_columns)

        card_columns = ["card1", "card2", "card3", "card5", "card6"]

        df["card_key"] = (
            df[card_columns]
            .astype("string")
            .fillna("missing")
            .agg("_".join, axis=1)
        )

        missing_card = df["card1"].isna()
        df.loc[missing_card, "card_key"] = (
            "missing_" + df.loc[missing_card, "TransactionID"].astype(str)
        )

        known = df[df["DeviceInfo"].notna()].copy()

        device_stats = known.groupby("DeviceInfo").agg(
            cards=("card_key", "nunique"),
            fraud_transactions=("isFraud", "sum"),
            transactions=("TransactionID", "count"),
        )

        selected = device_stats[
            device_stats["cards"].between(2, 20)
            & (device_stats["fraud_transactions"] > 0)
        ].nlargest(10, "fraud_transactions")

        if selected.empty:
            st.info("No qualifying shared-device groups were found.")
            st.stop()

        network_rows = known[
            known["DeviceInfo"].isin(selected.index)
        ].copy()

        card_stats = network_rows.groupby("card_key").agg(
            is_fraud=("isFraud", "max"),
            transactions=("TransactionID", "count"),
            fraud_transactions=("isFraud", "sum"),
        )

        edges = network_rows[
            ["DeviceInfo", "card_key"]
        ].drop_duplicates()

        graph = nx.Graph()

        for device, row in selected.iterrows():
            graph.add_node(
                ("device", device),
                kind="device",
                label=device,
                cards=int(row["cards"]),
                fraud_transactions=int(row["fraud_transactions"]),
                transactions=int(row["transactions"]),
            )

        for card, row in card_stats.iterrows():
            graph.add_node(
                ("card", card),
                kind="card",
                label=card,
                is_fraud=int(row["is_fraud"]),
                transactions=int(row["transactions"]),
                fraud_transactions=int(row["fraud_transactions"]),
            )

        for _, edge in edges.iterrows():
            graph.add_edge(
                ("device", edge["DeviceInfo"]),
                ("card", edge["card_key"]),
            )

        positions = nx.spring_layout(
            graph,
            seed=42,
            k=0.8,
            iterations=80,
        )

    fraud_cards = [
        node for node, attrs in graph.nodes(data=True)
        if attrs["kind"] == "card" and attrs["is_fraud"] == 1
    ]

    other_cards = [
        node for node, attrs in graph.nodes(data=True)
        if attrs["kind"] == "card" and attrs["is_fraud"] == 0
    ]

    device_nodes = [
        node for node, attrs in graph.nodes(data=True)
        if attrs["kind"] == "device"
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shared Devices", len(device_nodes))
    col2.metric("Connected Cards", len(card_stats))
    col3.metric("Card-Device Links", len(edges))
    col4.metric("Cards with Fraud Labels", len(fraud_cards))

    st.info(
        "Red = card has at least one confirmed fraud transaction in the "
        "training data. Grey = no confirmed fraud transaction in this data. "
        "Blue = shared device. A shared device is an investigation lead, "
        "not proof of a coordinated fraud ring."
    )

    edge_x = []
    edge_y = []

    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#9ca3af"),
        hoverinfo="skip",
        name="Card-device links",
    ))

    def add_card_trace(nodes, name, color):
        if not nodes:
            return

        fig.add_trace(go.Scatter(
            x=[positions[node][0] for node in nodes],
            y=[positions[node][1] for node in nodes],
            mode="markers",
            name=name,
            marker=dict(size=12, color=color, line=dict(width=1, color="white")),
            customdata=[
                [
                    graph.nodes[node]["label"],
                    graph.nodes[node]["transactions"],
                    graph.nodes[node]["fraud_transactions"],
                ]
                for node in nodes
            ],
            hovertemplate=(
                "Card: %{customdata[0]}<br>"
                "Transactions: %{customdata[1]}<br>"
                "Fraud transactions: %{customdata[2]}"
                "<extra></extra>"
            ),
        ))

    add_card_trace(fraud_cards, "Confirmed fraud card", "#dc2626")
    add_card_trace(other_cards, "Other connected card", "#9ca3af")

    fig.add_trace(go.Scatter(
        x=[positions[node][0] for node in device_nodes],
        y=[positions[node][1] for node in device_nodes],
        mode="markers",
        name="Shared device",
        marker=dict(
            size=20,
            color="#2563eb",
            symbol="diamond",
            line=dict(width=1, color="white"),
        ),
        customdata=[
            [
                graph.nodes[node]["label"],
                graph.nodes[node]["cards"],
                graph.nodes[node]["fraud_transactions"],
                graph.nodes[node]["transactions"],
            ]
            for node in device_nodes
        ],
        hovertemplate=(
            "Device: %{customdata[0]}<br>"
            "Connected cards: %{customdata[1]}<br>"
            "Fraud transactions: %{customdata[2]}<br>"
            "Total transactions: %{customdata[3]}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        height=720,
        title="Top 10 Shared Devices and Connected Cards",
        showlegend=True,
        hovermode="closest",
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected Shared Devices")
    st.dataframe(
        selected.reset_index(),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Graph uses confirmed IEEE-CIS training labels; "
        "it does not establish that connected accounts belong to the same actor."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AIML-01 Financial Fraud Detection | "
    "Streamlit Analyst Dashboard"
)

