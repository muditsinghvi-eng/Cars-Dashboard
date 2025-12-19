import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Config
st.set_page_config(layout="wide", page_title="Car Performance Dashboard")

try:
    import streamlit_analytics2 as streamlit_analytics
    analytics_available = True
except ImportError:
    analytics_available = False

# 2. Load Data
@st.cache_data(ttl=600)
def load_data():
    file_path = "Cars Dashboard Mobile.xlsx"
    sheet_mapping = {
        "search": "Search", "pv": "pv", "uu": "uu", 
        "mofu": "mofu", "leads": "leads", "gender": "gender", 
        "age": "Age", "reg_traffic": "Region", "reg_leads": "Lead Region"
    }
    data = {}
    try:
        for key, sheet in sheet_mapping.items():
            df = pd.read_excel(file_path, sheet_name=sheet)
            df.columns = [str(c).strip() for c in df.columns]
            name_col = 'model name' if 'model name' in df.columns else 'Model Name'
            if name_col in df.columns:
                df[name_col] = df[name_col].astype(str).str.strip()
                df = df.rename(columns={name_col: 'model name'})
            data[key] = df
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

data = load_data()

if analytics_available:
    streamlit_analytics.start_tracking()

# 3. Sidebar
all_models = sorted(data['search']['model name'].dropna().unique().tolist())
st.sidebar.header("Filters")
main_model = st.sidebar.selectbox("Main Model", all_models, index=0)
competitors = st.sidebar.multiselect("Competitors", [m for m in all_models if m != main_model])
months_list = ["Jan'25", "Feb'25", "Mar'25", "Apr'25", "May'25", "Jun'25", "Jul'25", "Aug'25", "Sep'25", "Oct'25", "Nov'25"]
selected_month = st.sidebar.selectbox("Month", months_list, index=10)

all_selected = [main_model] + competitors
s_idx = months_list.index(selected_month)
six_months = months_list[max(0, s_idx-5) : s_idx+1]
mobile_config = {'staticPlot': True}

st.title(f"Market Performance: {main_model}")

# --- SECTION 1: SOV TREND (Full Width Restored) ---
st.divider()
st.header("SOV % Trend")
sov_data = []
for m in six_months:
    row = {"Month": m}
    for metric in ["search", "pv", "uu", "mofu", "leads"]:
        df = data[metric]
        grp = df[df['model name'].isin(all_selected)]
        total = grp[m].sum()
        main_v = grp[grp['model name'] == main_model][m].sum()
        row[metric.upper()] = (main_v/total*100) if total > 0 else 0
    sov_data.append(row)

line_df = pd.DataFrame(sov_data).melt(id_vars="Month")
line_df['Label'] = line_df.apply(lambda x: f"<b>{x['value']:.1f}%</b>" if x['Month'] in [six_months[0], six_months[-1]] else "", axis=1)

fig_line = px.line(line_df, x="Month", y="value", color="variable", markers=True, text="Label")

# Resetting margins to allow full width and height
fig_line.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
    xaxis_title=None, 
    yaxis_title=None, 
    height=400, 
    margin=dict(l=20, r=20, t=50, b=20) 
)
fig_line.update_traces(textposition="top center")
st.plotly_chart(fig_line, use_container_width=True, config=mobile_config)

# --- SECTION 2: HEATMAPS (2-Column Layout) ---
st.divider()
st.header("Performance Heatmaps")
metrics = ["search", "pv", "uu", "mofu", "leads", "pv/uu", "t2l"]
h_cols = st.columns(2)
for i, m in enumerate(metrics):
    # (Heatmap logic remains the same to keep them compact)
    if m in ["pv/uu", "t2l"]:
        u_df = data["uu"][data["uu"]['model name'].isin(all_selected)].set_index('model name')[six_months]
        if m == "pv/uu":
            p_df = data["pv"][data["pv"]['model name'].isin(all_selected)].set_index('model name')[six_months]
            disp = (p_df / u_df).fillna(0); title, fmt = "PV/UU", ".2f"
        else:
            l_df = data["leads"][data["leads"]['model name'].isin(all_selected)].set_index('model name')[six_months]
            disp = ((l_df/u_df)*100).fillna(0); title, fmt = "T2L %", ".2f"
    else:
        raw = data[m][data[m]['model name'].isin(all_selected)].set_index('model name')[six_months]
        disp = (raw.div(raw.sum(), axis=1)*100).fillna(0); title, fmt = f"SOV: {m.upper()}", ".1f"
    
    with h_cols[i % 2]:
        fig = px.imshow(disp.reindex(all_selected), text_auto=fmt, color_continuous_scale="Reds")
        fig.update_layout(title=title, coloraxis_showscale=False, xaxis_title=None, yaxis_title=None, height=280, margin=dict(t=30, l=5, r=5))
        fig.update_traces(textfont=dict(weight='bold', size=14))
        st.plotly_chart(fig, use_container_width=True, config=mobile_config)

# --- SECTION 4: GEOGRAPHIC (Full Width Bars) ---
st.divider()
st.header("Top 10 States Performance wrt Traffic")

def get_geo_fig(key, main_m, comps, title, base_states=None):
    df = data[key]
    cols = [c for c in df.columns if c not in ['model name', 'Unnamed: 0', 'total', 'Total']]
    m_row = df[df['model name'] == main_m]
    if base_states is None:
        if m_row.empty: return None, []
        vals = m_row[cols].iloc[0]
        states = vals[vals > 0].sort_values(ascending=True).tail(10).index.tolist()
    else: states = [s for s in base_states if s in df.columns][::-1]
    
    p_data = []
    if not m_row.empty:
        t = m_row[cols].sum(axis=1).values[0]
        for s in states: p_data.append({"State": s, "Entity": main_m, "V": (m_row[s].values[0]/t*100 if t>0 else 0)})
    
    c_rows = df[df['model name'].isin(comps)]
    if not c_rows.empty:
        avgs = c_rows[cols].mean(); t = avgs.sum()
        for s in states: p_data.append({"State": s, "Entity": "Comp Avg", "V": (avgs[s]/t*100 if t>0 else 0)})
    
    if not p_data: return None, []
    fig = px.bar(pd.DataFrame(p_data), y="State", x="V", color="Entity", barmode="group", text_auto='.1f', orientation='h', title=title)
    # RESTORED WIDTH for bars
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=1), xaxis_title=None, yaxis_title=None, height=500, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_traces(textfont=dict(weight='bold'))
    return fig, states[::-1]

# We no longer use columns(2) for the state bars to prevent squeezing
fig_t, t_states = get_geo_fig('reg_traffic', main_model, competitors, "Traffic Contribution %")
if fig_t: st.plotly_chart(fig_t, use_container_width=True, config=mobile_config)

fig_l, _ = get_geo_fig('reg_leads', main_model, competitors, "Leads Contribution %", base_states=t_states)
if fig_l: st.plotly_chart(fig_l, use_container_width=True, config=mobile_config)

# --- SECTION 5: T2L HEATMAP (Full Width) ---
st.subheader("State T2L %")
if t_states:
    valid = [s for s in t_states if s in data['reg_traffic'].columns and s in data['reg_leads'].columns]
    if valid:
        t_sub = data['reg_traffic'][data['reg_traffic']['model name'].isin(all_selected)].set_index('model name')[valid]
        l_sub = data['reg_leads'][data['reg_leads']['model name'].isin(all_selected)].set_index('model name')[valid]
        t2l = (l_sub/t_sub*100).fillna(0).T
        fig_z = px.imshow(t2l, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto")
        fig_z.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None, height=550, margin=dict(l=20, r=20, t=20, b=20))
        fig_z.update_traces(textfont=dict(weight='bold', size=12))
        st.plotly_chart(fig_z, use_container_width=True, config=mobile_config)

if analytics_available:
    streamlit_analytics.stop_tracking()