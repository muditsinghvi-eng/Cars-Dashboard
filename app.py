import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="Car SOV Dashboard")

# 1. Load Data from a single Excel file
@st.cache_data
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
            data[key] = pd.read_excel(file_path, sheet_name=sheet)
        return data
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()

data = load_data()

# Clean up model names and column names
for key in data:
    name_col = 'model name' if 'model name' in data[key].columns else 'Model Name'
    if name_col in data[key].columns:
        data[key][name_col] = data[key][name_col].astype(str).str.strip()
    if key in ['reg_traffic', 'reg_leads']:
        data[key].columns = [str(col).strip() for col in data[key].columns]

all_models = sorted(data['search']['model name'].dropna().unique().tolist())

# Sidebar
st.sidebar.header("Dashboard Filters")
main_model = st.sidebar.selectbox("Select Main Model", all_models, index=all_models.index("Mahindra XUV400 EV") if "Mahindra XUV400 EV" in all_models else 0)
competitors = st.sidebar.multiselect("Select Competitors", [m for m in all_models if m != main_model], max_selections=10)

# Month Handling
months_list = ["Jan'25", "Feb'25", "Mar'25", "Apr'25", "May'25", "Jun'25", "Jul'25", "Aug'25", "Sep'25", "Oct'25", "Nov'25"]
selected_month = st.sidebar.selectbox("Select Month", months_list, index=10)

# Calculate the last 6 months range
selected_idx = months_list.index(selected_month)
start_idx = max(0, selected_idx - 5)
six_months_window = months_list[start_idx : selected_idx + 1]

all_selected = [main_model] + competitors

st.title(f"Market Share & Performance Analysis: {main_model}")

# --- SOV TREND LINE CHART ---
st.divider()
st.header(f"📈 Funnel SOV % Trend: {main_model}")
funnel_metrics = ["search", "pv", "uu", "mofu", "leads"]
sov_trend_data = []
for month in six_months_window:
    row = {"Month": month}
    for m in funnel_metrics:
        df = data[m]
        group_df = df[df['model name'].isin(all_selected)]
        total_val = group_df[month].sum()
        main_val = group_df[group_df['model name'] == main_model][month].sum()
        row[m.upper()] = (main_val / total_val * 100) if total_val > 0 else 0
    sov_trend_data.append(row)
line_plot_df = pd.DataFrame(sov_trend_data).melt(id_vars="Month", var_name="Metric", value_name="SOV %")
fig_line = px.line(line_plot_df, x="Month", y="SOV %", color="Metric", markers=True, title=f"SOV % Trend for {main_model}")
st.plotly_chart(fig_line, use_container_width=True, key="sov_line")

# --- PERFORMANCE HEATMAPS ---
st.divider()
st.header("📊 6-Month Performance Heatmaps")
metrics_list = ["search", "pv", "uu", "mofu", "leads", "pv/uu", "t2l"]
cols = st.columns(2)
for i, m in enumerate(metrics_list):
    if m in ["pv/uu", "t2l"]:
        df_u = data["uu"][data["uu"]['model name'].isin(all_selected)].set_index('model name')[six_months_window]
        if m == "pv/uu":
            df_p = data["pv"][data["pv"]['model name'].isin(all_selected)].set_index('model name')[six_months_window]
            display_df = (df_p / df_u).fillna(0)
            title, fmt = "Engagement Depth (PV/UU)", ".2f"
        else:
            df_l = data["leads"][data["leads"]['model name'].isin(all_selected)].set_index('model name')[six_months_window]
            display_df = ((df_l / df_u) * 100).fillna(0)
            title, fmt = "Lead Conversion Rate %", ".2f"
    else:
        df = data[m]
        raw_v = df[df['model name'].isin(all_selected)].set_index('model name')[six_months_window]
        display_df = (raw_v.div(raw_v.sum(axis=0), axis=1) * 100).fillna(0)
        title, fmt = f"SOV %: {m.upper()}", ".2f"
    display_df = display_df.reindex(all_selected)
    with cols[i % 2]:
        st.plotly_chart(px.imshow(display_df, text_auto=fmt, aspect="auto", color_continuous_scale="YlGnBu", title=title), use_container_width=True, key=f"heat_{m}")

# --- AUDIENCE COMPOSITION ---
st.divider()
st.header("👥 Audience Profile: Composition % Analysis")
col_g, col_a = st.columns(2)
with col_g:
    g_df = data['gender'][data['gender']['model name'].isin(all_selected)]
    main_g, comp_g = g_df[g_df['model name'] == main_model], g_df[g_df['model name'].isin(competitors)]
    g_list = []
    if not main_g.empty:
        t = main_g[['Male','Female']].sum(axis=1).values[0]
        g_list.append({"Entity": main_model, "Male %": main_g['Male'].values[0]/t*100, "Female %": main_g['Female'].values[0]/t*100})
    if not comp_g.empty:
        m, f = comp_g['Male'].mean(), comp_g['Female'].mean()
        g_list.append({"Entity": "Competition (Avg)", "Male %": m/(m+f)*100, "Female %": f/(m+f)*100})
    st.plotly_chart(px.bar(pd.DataFrame(g_list).melt(id_vars="Entity"), x="Entity", y="value", color="variable", barmode="group", text_auto='.1f', title="Gender Split %"), use_container_width=True, key="g_comp")
with col_a:
    age_cols = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    a_df = data['age'][data['age']['model name'].isin(all_selected)]
    main_a, comp_a = a_df[a_df['model name'] == main_model], a_df[a_df['model name'].isin(competitors)]
    a_list = []
    if not main_a.empty:
        t = main_a[age_cols].sum(axis=1).values[0]
        row = {"Entity": main_model}
        for c in age_cols: row[c] = main_a[c].values[0]/t*100
        a_list.append(row)
    if not comp_a.empty:
        avgs = comp_a[age_cols].mean(); t = avgs.sum()
        row = {"Entity": "Competition (Avg)"}
        for c in age_cols: row[c] = avgs[c]/t*100
        a_list.append(row)
    st.plotly_chart(px.bar(pd.DataFrame(a_list).melt(id_vars="Entity"), x="variable", y="value", color="Entity", barmode="group", text_auto='.1f', title="Age Split %"), use_container_width=True, key="a_comp")

# --- GEOGRAPHIC DISTRIBUTION ---
st.divider()
st.header("📍 Geographic Performance: Top 10 States for Main Model")

def prepare_geo_comp_plot(df_key, main_m, comp_list, title, baseline_states=None):
    df = data[df_key]
    m_col = 'model name' if 'model name' in df.columns else 'Model Name'
    exclude = [m_col, 'Unnamed: 0']
    state_cols = [c for c in df.columns if c not in exclude]
    main_row = df[df[m_col] == main_m][state_cols]
    comp_rows = df[df[m_col].isin(comp_list)][state_cols]
    
    if baseline_states is None:
        m_total = main_row.sum(axis=1).values[0] if not main_row.empty else 1
        selected_states = (main_row.iloc[0] / m_total * 100).sort_values(ascending=False).head(10).index.tolist()
    else:
        selected_states = baseline_states

    plot_data = []
    m_total = main_row.sum(axis=1).values[0] if not main_row.empty else 1
    for s in selected_states:
        val = main_row[s].values[0] if s in main_row.columns else 0
        plot_data.append({"State": s, "Entity": main_m, "Composition %": (val / m_total * 100)})
    if not comp_rows.empty:
        avg_profile = comp_rows.mean()
        avg_total = avg_profile.sum()
        for s in selected_states:
            val = avg_profile[s] if s in avg_profile.index else 0
            plot_data.append({"State": s, "Entity": "Competition (Avg)", "Composition %": (val / avg_total * 100)})
    return px.bar(pd.DataFrame(plot_data), x="State", y="Composition %", color="Entity", barmode="group", text_auto='.1f', title=title), selected_states

geo_col1, geo_col2 = st.columns(2)
fig_geo_t, top_states = prepare_geo_comp_plot('reg_traffic', main_model, competitors, "Top 10 States: Traffic (UU) % Contribution")
fig_geo_l, _ = prepare_geo_comp_plot('reg_leads', main_model, competitors, "Top 10 States: Leads % Contribution", baseline_states=top_states)
with geo_col1: st.plotly_chart(fig_geo_t, use_container_width=True, key="geo_t_pct")
with geo_col2: st.plotly_chart(fig_geo_l, use_container_width=True, key="geo_l_pct")

# --- NEW: GEOGRAPHIC T2L HEATMAP ---
st.subheader("🎯 Geographic Conversion (T2L %) Heatmap")
st.write(f"This matrix shows the conversion efficiency (Leads/UU) across the top 10 states for all selected models.")

m_col_geo = 'model name' if 'model name' in data['reg_traffic'].columns else 'Model Name'
t_subset = data['reg_traffic'][data['reg_traffic'][m_col_geo].isin(all_selected)].set_index(m_col_geo)[top_states]
l_subset = data['reg_leads'][data['reg_leads'][m_col_geo].isin(all_selected)].set_index(m_col_geo)[top_states]

# Calculate T2L % for each model per state
t2l_geo_df = (l_subset / t_subset * 100).fillna(0).reindex(all_selected)

fig_t2l_heat = px.imshow(
    t2l_geo_df,
    labels=dict(x="State", y="Model", color="T2L %"),
    x=t2l_geo_df.columns,
    y=t2l_geo_df.index,
    text_auto=".2f",
    color_continuous_scale="RdYlGn", # Red for low conversion, Green for high
    aspect="auto"
)
st.plotly_chart(fig_t2l_heat, use_container_width=True, key="geo_t2l_heat")