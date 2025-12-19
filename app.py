import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="Car SOV Dashboard")

# 1. Load Data
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

# Clean up
for key in data:
    name_col = 'model name' if 'model name' in data[key].columns else 'Model Name'
    if name_col in data[key].columns:
        data[key][name_col] = data[key][name_col].astype(str).str.strip()
    if key in ['reg_traffic', 'reg_leads']:
        # Ensure all column names are strings and stripped of hidden spaces
        data[key].columns = [str(col).strip() for col in data[key].columns]

all_models = sorted(data['search']['model name'].dropna().unique().tolist())

# Sidebar
st.sidebar.header("Dashboard Filters")
main_model = st.sidebar.selectbox("Select Main Model", all_models, index=all_models.index("Mahindra XUV400 EV") if "Mahindra XUV400 EV" in all_models else 0)
competitors = st.sidebar.multiselect("Select Competitors", [m for m in all_models if m != main_model], max_selections=10)
months_list = ["Jan'25", "Feb'25", "Mar'25", "Apr'25", "May'25", "Jun'25", "Jul'25", "Aug'25", "Sep'25", "Oct'25", "Nov'25"]
selected_month = st.sidebar.selectbox("Select Month", months_list, index=10)

# 6-Month Window
selected_idx = months_list.index(selected_month)
start_idx = max(0, selected_idx - 5)
six_months_window = months_list[start_idx : selected_idx + 1]
all_selected = [main_model] + competitors

st.title(f"Market Performance: {main_model}")

mobile_config = {'displayModeBar': False, 'staticPlot': False}

# --- 1. SOV TREND LINE CHART ---
st.divider()
st.header(f"📈 Funnel SOV % Trend")
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
line_plot_df['Label'] = line_plot_df.apply(
    lambda x: f"<b>{x['SOV %']:.1f}%</b>" if x['Month'] in [six_months_window[0], six_months_window[-1]] else "", 
    axis=1
)

fig_line = px.line(line_plot_df, x="Month", y="SOV %", color="Metric", markers=True, text="Label")
fig_line.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
    xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=30, b=0), height=350
)
fig_line.update_traces(textposition="top center")
st.plotly_chart(fig_line, use_container_width=True, config=mobile_config, key="sov_line")

# --- 2. PERFORMANCE HEATMAPS ---
st.divider()
st.header("📊 6-Month Detailed Heatmaps")
metrics_list = ["search", "pv", "uu", "mofu", "leads", "pv/uu", "t2l"]
cols = st.columns(2)

for i, m in enumerate(metrics_list):
    if m in ["pv/uu", "t2l"]:
        df_u = data["uu"][data["uu"]['model name'].isin(all_selected)].set_index('model name')[six_months_window]
        if m == "pv/uu":
            df_p = data["pv"][data["pv"]['model name'].isin(all_selected)].set_index('model name')[six_months_window]
            display_df = (df_p / df_u).fillna(0)
            title, fmt = "Depth (PV/UU)", ".2f"
        else:
            df_l = data["leads"][data["leads"]['model name'].isin(all_selected)].set_index('model name')[six_months_window]
            display_df = ((df_l / df_u) * 100).fillna(0)
            title, fmt = "Conv Rate %", ".2f"
    else:
        df = data[m]
        raw_v = df[df['model name'].isin(all_selected)].set_index('model name')[six_months_window]
        display_df = (raw_v.div(raw_v.sum(axis=0), axis=1) * 100).fillna(0)
        title, fmt = f"SOV: {m.upper()}", ".1f"
    
    display_df = display_df.reindex(all_selected)
    with cols[i % 2]:
        fig_heat = px.imshow(display_df, text_auto=fmt, aspect="auto", color_continuous_scale="Reds", title=title)
        fig_heat.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None, margin=dict(l=5, r=5, t=35, b=5), height=280)
        fig_heat.update_traces(textfont=dict(size=14, color="black", weight='bold')) 
        st.plotly_chart(fig_heat, use_container_width=True, config=mobile_config, key=f"heat_{m}")

# --- 3. AUDIENCE COMPOSITION ---
st.divider()
st.header("👥 Audience Profile %")
col_g, col_a = st.columns(2)
bar_layout = dict(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None), xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=35, b=0), height=300)

with col_g:
    g_df = data['gender'][data['gender']['model name'].isin(all_selected)]
    main_g, comp_g = g_df[g_df['model name'] == main_model], g_df[g_df['model name'].isin(competitors)]
    g_list = []
    if not main_g.empty:
        t = main_g[['Male','Female']].sum(axis=1).values[0]
        if t > 0:
            g_list.append({"Entity": main_model, "Male %": main_g['Male'].values[0]/t*100, "Female %": main_g['Female'].values[0]/t*100})
    if not comp_g.empty:
        m, f = comp_g['Male'].mean(), comp_g['Female'].mean()
        if (m+f) > 0:
            g_list.append({"Entity": "Comp Avg", "Male %": m/(m+f)*100, "Female %": f/(m+f)*100})
    if g_list:
        fig_g = px.bar(pd.DataFrame(g_list).melt(id_vars="Entity"), x="Entity", y="value", color="variable", barmode="group", text_auto='.1f', title="Gender Split %")
        fig_g.update_layout(bar_layout)
        fig_g.update_traces(textfont=dict(weight='bold', size=13))
        st.plotly_chart(fig_g, use_container_width=True, config=mobile_config, key="g_comp")

with col_a:
    age_cols = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    a_df = data['age'][data['age']['model name'].isin(all_selected)]
    main_a, comp_a = a_df[a_df['model name'] == main_model], a_df[a_df['model name'].isin(competitors)]
    a_list = []
    if not main_a.empty:
        t = main_a[age_cols].sum(axis=1).values[0]
        if t > 0:
            row = {"Entity": main_model}
            for c in age_cols: row[c] = main_a[c].values[0]/t*100
            a_list.append(row)
    if not comp_a.empty:
        avgs = comp_a[age_cols].mean(); t = avgs.sum()
        if t > 0:
            row = {"Entity": "Comp Avg"}
            for c in age_cols: row[c] = avgs[c]/t*100
            a_list.append(row)
    if a_list:
        fig_a = px.bar(pd.DataFrame(a_list).melt(id_vars="Entity"), x="variable", y="value", color="Entity", barmode="group", text_auto='.1f', title="Age Split %")
        fig_a.update_layout(bar_layout)
        fig_a.update_traces(textfont=dict(weight='bold', size=13))
        st.plotly_chart(fig_a, use_container_width=True, config=mobile_config, key="a_comp")

# --- 4. GEOGRAPHIC PERFORMANCE (FIXED KEYERROR & CHANDIGARH ISSUE) ---
st.divider()
st.header("📍 Top 10 States Performance")

def prepare_geo_plot(df_key, main_m, comp_list, title, baseline_states=None):
    df = data[df_key]
    m_col = 'model name' if 'model name' in df.columns else 'Model Name'
    
    # Identify state columns excluding model name and other technical columns
    state_cols = [c for c in df.columns if c not in [m_col, 'Unnamed: 0', 'total', 'Total']]
    
    # Extract row for main model
    main_row = df[df[m_col] == main_m]
    
    if baseline_states is None:
        # Determine Top 10 based on this specific sheet if no baseline is provided
        if not main_row.empty:
            vals = main_row[state_cols].iloc[0]
            # Filter out zero values to avoid "Chandigarh" issues if it has 0 traffic
            sel_states = vals[vals > 0].sort_values(ascending=True).tail(10).index.tolist()
        else:
            sel_states = []
    else: 
        # Use baseline states but reverse for top-to-bottom bar chart
        sel_states = baseline_states[::-1]
        
    plot_data = []
    # Calculate percentages
    if not main_row.empty:
        m_total = main_row[state_cols].sum(axis=1).values[0]
        for s in sel_states:
            # Check if state exists in this specific sheet (fixes Ladakh KeyError)
            val = main_row[s].values[0] if s in main_row.columns else 0
            pct = (val / m_total * 100) if m_total > 0 else 0
            plot_data.append({"State": s, "Entity": main_m, "Val": pct})
    
    comp_rows = df[df[m_col].isin(comp_list)]
    if not comp_rows.empty:
        avg_profile = comp_rows[state_cols].mean()
        avg_total = avg_profile.sum()
        for s in sel_states:
            val = avg_profile[s] if s in avg_profile.index else 0
            pct = (val / avg_total * 100) if avg_total > 0 else 0
            plot_data.append({"State": s, "Entity": "Comp Avg", "Val": pct})
    
    if plot_data:
        fig = px.bar(pd.DataFrame(plot_data), y="State", x="Val", color="Entity", 
                     barmode="group", text_auto='.1f', title=title, orientation='h')
        fig.update_traces(textfont=dict(weight='bold', size=12))
        return fig, sel_states[::-1]
    return None, []

geo_col1, geo_col2 = st.columns(2)
fig_t, t_states = prepare_geo_plot('reg_traffic', main_model, competitors, "Traffic Contribution %")
fig_l, _ = prepare_geo_plot('reg_leads', main_model, competitors, "Leads Contribution %", baseline_states=t_states)

for f in [fig_t, fig_l]: 
    if f: f.update_layout(bar_layout, height=450)

with geo_col1:
    if fig_t: st.plotly_chart(fig_t, use_container_width=True, config=mobile_config, key="geo_t")
with geo_col2:
    if fig_l: st.plotly_chart(fig_l, use_container_width=True, config=mobile_config, key="geo_l")

# Geographic T2L Heatmap
st.subheader("🎯 Geographic T2L % Heatmap")
if t_states:
    m_col_geo = 'model name' if 'model name' in data['reg_traffic'].columns else 'Model Name'
    # Safely select columns that exist in both sheets
    valid_states = [s for s in t_states if s in data['reg_traffic'].columns and s in data['reg_leads'].columns]
    
    if valid_states:
        t_sub = data['reg_traffic'][data['reg_traffic'][m_col_geo].isin(all_selected)].set_index(m_col_geo)[valid_states]
        l_sub = data['reg_leads'][data['reg_leads'][m_col_geo].isin(all_selected)].set_index(m_col_geo)[valid_states]
        t2l_geo = (l_sub / t_sub * 100).fillna(0).reindex(all_selected)

        fig_t2l = px.imshow(t2l_geo, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto")
        fig_t2l.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None, margin=dict(l=5, r=5, t=10, b=5), height=280)
        fig_t2l.update_traces(textfont=dict(size=14, color="black", weight='bold'))
        st.plotly_chart(fig_t2l, use_container_width=True, config=mobile_config, key="geo_t2l")