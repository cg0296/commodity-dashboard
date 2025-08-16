import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import datetime as dt

# =========================================================
# Data (dummy data for two commodities: CSC, ZC)
# =========================================================
commodity_data = {
    "CSC": {
        "metric": [
            "Contract Code", "Current Fwd Price", "Insurance Premium",
            "Predicted Spot Price", "Predicted Insurance Premium",
            "Premium Spread", "Predicted Spot Price Direction",
            "Predicted Spot Price % Move", "Predicted Forward Variance"
        ],
        "Jul-2025": ["BJQ25", 1.816, 0.000, 1.750, -0.066, 0.066, -1.0, "-6.6%", -0.066],
        "Aug-2025": ["BJU25", 1.759, 0.043, 1.780, -0.036, 0.079, -1.0, "-3.6%", -0.079],
        "Sep-2025": ["BJV25", 1.903, 0.087, 1.930, 0.114, -0.027, 1.0, "11.4%", 0.027]
    },
    "ZC": {
        "metric": [
            "Contract Code", "Current Fwd Price", "Insurance Premium",
            "Predicted Spot Price", "Predicted Insurance Premium",
            "Premium Spread", "Predicted Spot Price Direction",
            "Predicted Spot Price % Move", "Predicted Forward Variance"
        ],
        "Jul-2025": ["ZCQ25", 4.750, 0.100, 4.800, -0.050, 0.150, 1.0, "1.0%", -0.050],
        "Aug-2025": ["ZCU25", 4.880, 0.120, 4.850, -0.030, 0.150, 1.0, "0.5%", -0.030],
        "Sep-2025": ["ZCV25", 4.910, 0.130, 4.890, 0.010, 0.120, -1.0, "-0.4%", 0.010]
    }
}

# =========================================================
# Utilities
# =========================================================
PERCENT_METRICS = {"Predicted Spot Price % Move"}
ALL_KPI_METRICS = [
    "Current Fwd Price",
    "Predicted Spot Price",
    "Insurance Premium",
    "Predicted Insurance Premium",
    "Premium Spread",
    "Predicted Spot Price Direction",
    "Predicted Spot Price % Move",
    "Predicted Forward Variance",
]

def get_df(comm: str) -> pd.DataFrame:
    return pd.DataFrame(commodity_data[comm])

def extract_numeric_series(df: pd.DataFrame, row_label: str) -> list[float]:
    raw = df[df["metric"] == row_label].iloc[:, 1:].values.flatten()
    out = []
    for v in raw:
        if isinstance(v, (int, float, np.number)):
            out.append(float(v))
        elif isinstance(v, str):
            s = v.strip()
            if s.endswith("%"):
                try:
                    out.append(float(s[:-1]) / 100.0)
                except Exception:
                    out.append(np.nan)
            else:
                try:
                    out.append(float(s))
                except Exception:
                    out.append(np.nan)
        else:
            out.append(np.nan)
    return out

def build_line(x, y, name, color=None, hover="Price: %{y:.3f}"):
    return go.Scatter(
        x=x, y=y, mode="lines+markers", name=name,
        hovertemplate=f"<b>%{{x}}</b><br>{hover}<extra></extra>",
        line=dict(color=color) if color else None
    )

def compute_kpi_for_metric(df: pd.DataFrame, metric_name: str):
    y = extract_numeric_series(df, metric_name)
    if len(y) < 2 or any(pd.isna(y[:2])):
        return (metric_name, "N/A", "", "#6b7280")
    base, nxt = float(y[0]), float(y[1])
    delta = nxt - base
    color = "#22c55e" if delta > 0 else "#ef4444" if delta < 0 else "#6b7280"
    if metric_name in PERCENT_METRICS:
        abs_txt = f"{delta*100:+.1f}%"
        pct_txt = ""
    elif metric_name == "Predicted Spot Price Direction":
        abs_txt = f"{delta:+.0f}"
        pct_txt = ""
    else:
        abs_txt = f"{delta:+.3f}"
        pct_txt = ""
    return (metric_name, abs_txt, pct_txt, color)

def make_kpi_card(title: str, abs_txt: str, pct_txt: str, color: str):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),
            html.H2(abs_txt, style={"color": color, "margin": 0}),
            (html.Div(pct_txt, style={"color": color, "fontSize": "1.0rem"}) if pct_txt else html.Div())
        ]),
        className="shadow-sm rounded-4",
        style={"width": "100%"}
    )

def build_kpi_row(df: pd.DataFrame):
    cards = []
    for m in ALL_KPI_METRICS:
        title, abs_txt, pct_txt, color = compute_kpi_for_metric(df, m)
        cards.append(dbc.Col(make_kpi_card(title, abs_txt, pct_txt, color), md=3))
    return dbc.Row(cards, className="g-3")

def build_figures(df: pd.DataFrame):
    months = list(df.columns[1:])
    y_pred_spot = extract_numeric_series(df, "Predicted Spot Price")
    y_fwd_price = extract_numeric_series(df, "Current Fwd Price")

    fig_spot = go.Figure()
    fig_spot.add_trace(build_line(months, y_pred_spot, "Predicted Spot Price"))
    fig_spot.update_layout(template="plotly_white", title="Predicted Spot Price",
                           xaxis_title="Month", yaxis_title="Price")

    fig_cmp = go.Figure()
    fig_cmp.add_trace(build_line(months, y_fwd_price, "Current Fwd Price"))
    fig_cmp.add_trace(build_line(months, y_pred_spot, "Predicted Spot Price"))
    fig_cmp.update_layout(template="plotly_white", title="Forward vs Predicted",
                          xaxis_title="Month", yaxis_title="Price")

    return fig_spot, fig_cmp

# ---------- Forecast helpers ----------
def build_forecast_series(comm: str, today: dt.date = None):
    rng_today = today or dt.date.today()
    past_months = pd.date_range(rng_today - pd.DateOffset(months=17), periods=18, freq="MS")
    fut_months = pd.date_range(rng_today + pd.DateOffset(months=1), periods=12, freq="MS")
    base = 1.8 if comm == "CSC" else 4.8
    hist = base + 0.15*np.sin(np.linspace(0, 3.5, len(past_months))) + 0.05*np.random.RandomState(7).randn(len(past_months))
    fut = np.linspace(hist[-1], hist[-1] + 0.35, 6).tolist() + np.linspace(hist[-1] + 0.35, hist[-1] - 0.25, 6).tolist()
    fut = np.array(fut)
    return past_months, hist, fut_months, fut

def build_forecast_figure(comm: str):
    pm, hist, fm, fut = build_forecast_series(comm)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pm, y=hist, mode="lines+markers", name="History"))
    fig.add_trace(go.Scatter(x=fm, y=fut, mode="lines+markers", name="Forecast", line=dict(dash="dot")))
    labels = ["20", "21", "22", "23", "24"]
    for i, lab in enumerate(labels):
        if i < len(fm):
            fig.add_annotation(x=fm[i], y=fut[i], text=lab, showarrow=False,
                               bgcolor="#fff", bordercolor="#bbb", borderwidth=1, yshift=18)
    fig.update_layout(template="plotly_white", title="Forecast View",
                      xaxis_title="Month", yaxis_title="Price",
                      legend=dict(orientation="h", y=1.1, x=0), margin=dict(l=30, r=20, t=60, b=40))
    return fig

def forecast_kpis(comm: str):
    pm, hist, fm, fut = build_forecast_series(comm)
    current = hist[-1]
    m3 = fut[min(2, len(fut)-1)]
    m6 = fut[min(5, len(fut)-1)]
    kpi3 = (m3 - current) / current * 100
    kpi6 = (m6 - current) / current * 100
    return current, kpi3, kpi6, fm[2].date() if len(fm) > 2 else None, fm[5].date() if len(fm) > 5 else None

# =========================================================
# App
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server
app.title = "Commodity Dashboard"

df_default = get_df("CSC")
metric_options = df_default["metric"].tolist()

# ---------- Sidebar (sticky) ----------
SIDEBAR_STYLE = {
    "position": "sticky",
    "top": "1rem",
}

sidebar = dbc.Card(
    dbc.CardBody([
        html.H5("Controls", className="mb-3"),
        html.Label("Commodity"),
        dcc.Dropdown(
            id="commodity-select",
            options=[{"label": "CSC", "value": "CSC"}, {"label": "ZC", "value": "ZC"}],
            value="CSC",
            clearable=False,
            className="mb-3"
        ),
        html.Label("Metric filter"),
        dcc.Dropdown(
            id="metric-filter",
            options=[{"label": m, "value": m} for m in metric_options],
            multi=True,
            placeholder="Filter metrics…",
            className="mb-3"
        ),
        html.Label("Forecast date"),
        dcc.DatePickerSingle(id="forecast-date", date=dt.date.today(), className="mb-1"),
        html.Small("This drives the note on the Forecast tab.", className="text-muted"),
    ]),
    className="shadow-sm",
    style=SIDEBAR_STYLE
)

# ---------- Page 2 (Forward + Table) ----------
page2 = html.Div([
    html.Br(),
    html.Div(id="kpi-row"),
    html.Br(),
    dcc.Graph(id="cmp-chart", config={"displayModeBar": True}),
    html.Hr(),
    dash_table.DataTable(
        id="metrics-table",
        columns=[{"name": c, "id": c, "editable": (c != "metric")} for c in df_default.columns],
        data=df_default.to_dict("records") + [{"metric": "User Input", **{c: None for c in df_default.columns if c != "metric"}}],
        editable=True,
        row_deletable=True,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "6px"},
    ),
    html.Div(id="user-kpi", className="mt-2")
])

# ---------- Page 3 (Forecast) ----------
hedge_levels = ["Avoid", "Plan", "Partial", "Full"]

page3_sidebar = dbc.Card(
    dbc.CardBody([
        html.H5("Key Data", className="mb-3"),
        dbc.Badge("Downtrend", color="danger", className="mb-3"),
        html.Div("Buyer's Hedging Recommendation", className="fw-semibold mb-1"),
        dbc.ButtonGroup([
            dbc.Button(l, id={"type": "hedge-btn", "level": l}, outline=True, color="secondary")
            for l in hedge_levels
        ], className="mb-3"),
        html.Div(id="forecast-note", className="small text-muted"),
        html.Hr(),
        html.Div(id="forecast-kpis")
    ]),
    className="shadow-sm"
)

page3 = html.Div([
    html.Br(),
    dbc.Row([
        dbc.Col(dcc.Graph(id="forecast-fig", config={"displayModeBar": True}), md=9),
        dbc.Col(page3_sidebar, md=3),
    ])
])

# ---------- Spot page ----------
page1 = html.Div([
    html.Br(),
    dcc.Graph(id="spot-chart", config={"displayModeBar": True})
])

# ---------- Main layout: sidebar + content tabs ----------
app.layout = dbc.Container(fluid=True, children=[
    html.Br(),
    dbc.Row([
        dbc.Col(sidebar, md=3),
        dbc.Col(
            dcc.Tabs([
                dcc.Tab(label="Spot Price Chart", children=[page1]),
                dcc.Tab(label="Forward vs Predicted + Table", children=[page2]),
                dcc.Tab(label="Forecast", children=[page3]),
            ]),
            md=9
        )
    ])
])

# =========================================================
# Callbacks
# =========================================================
@app.callback(
    Output("kpi-row", "children"),
    Output("metrics-table", "data"),
    Output("spot-chart", "figure"),
    Output("cmp-chart", "figure"),
    Output("user-kpi", "children"),
    Input("metric-filter", "value"),
    Input("commodity-select", "value"),
    State("metrics-table", "data")
)
def update_content(selected_metrics, commodity, table_data):
    df_new = get_df(commodity)
    kpi_children = build_kpi_row(df_new)
    fig_spot, fig_cmp = build_figures(df_new)

    filtered = df_new if not selected_metrics else df_new[df_new["metric"].isin(selected_metrics)]
    user_rows = [r for r in (table_data or []) if r.get("metric") == "User Input"]
    data_out = filtered.to_dict("records") + user_rows

    user_kpi_div = html.Div()
    if user_rows:
        user_row = user_rows[0]
        months = list(df_new.columns[1:])
        try:
            base = float(df_new[df_new["metric"] == "Current Fwd Price"].iloc[0, 1])
            nxt = float(user_row.get(months[0]) or 0)
            delta = nxt - base
            color = "#22c55e" if delta > 0 else "#ef4444" if delta < 0 else "#6b7280"
            user_kpi_div = make_kpi_card("User vs Current Fwd", f"{delta:+.3f}", "", color)
        except Exception:
            pass

    return kpi_children, data_out, fig_spot, fig_cmp, user_kpi_div

@app.callback(
    Output("forecast-fig", "figure"),
    Output("forecast-note", "children"),
    Output("forecast-kpis", "children"),
    Input("commodity-select", "value"),
    Input("forecast-date", "date"),
    Input({"type":"hedge-btn","level":dash.ALL}, "n_clicks"),
)
def update_forecast(comm, date_selected, hedge_clicks):
    fig = build_forecast_figure(comm or "CSC")
    cur, k3, k6, d3, d6 = forecast_kpis(comm or "CSC")
    k3_color = "#22c55e" if k3 > 0 else "#ef4444" if k3 < 0 else "#6b7280"
    k6_color = "#22c55e" if k6 > 0 else "#ef4444" if k6 < 0 else "#6b7280"
    note = (f"{pd.to_datetime(date_selected).date() if date_selected else pd.Timestamp.today().date()}:"
            f" The price remains in a downtrend, but upside risk persists in the medium term.")
    cards = dbc.Row([
        dbc.Col(make_kpi_card("Next 3 months", f"{k3:+.2f}%", "", k3_color), md=6),
        dbc.Col(make_kpi_card("Next 6 months", f"{k6:+.2f}%", "", k6_color), md=6),
    ], className="g-2")
    return fig, note, cards

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
