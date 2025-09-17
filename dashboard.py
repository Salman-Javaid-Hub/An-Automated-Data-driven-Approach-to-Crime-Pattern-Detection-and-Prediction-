import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import networkx as nx
from prophet import Prophet
from fpdf import FPDF
import os
import matplotlib.ticker as mticker  # formatting for axes

st.set_page_config(page_title="Crime Pattern Dashboard", layout="wide")
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    df = pd.read_csv("crime_data.csv")
    df = df.rename(columns={
        "Offence Group": "OffenseGroup",
        "Force Name": "PoliceForce",
        "Number of Offences": "CrimeCount",
        "Financial Year": "Year"
    })

    # Normalize types
    df["Year"] = df["Year"].astype(str).str[:4].astype(int)
    df["CrimeCount"] = pd.to_numeric(df["CrimeCount"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data
def load_gm_map():
    shapefile_path = r"Police_Forces_UK.shp"
    gm_map = gpd.read_file(shapefile_path)
    gm_map = gm_map[gm_map['PFA24NM'].str.contains("Greater Manchester", case=False, na=False)]
    return gm_map

df = load_data()
gm_map = load_gm_map()

tab_dashboard, tab_export = st.tabs(["📊 Dashboard", "📄 Export Report"])

with tab_dashboard:
    st.sidebar.header("🔍 Filters")
    crime_types = sorted(df["OffenseGroup"].dropna().unique())
    years = sorted(df["Year"].dropna().unique())

    select_all = st.sidebar.checkbox("Select All Offenses", value=True)
    if select_all:
        crime_filter = crime_types
    else:
        crime_filter = st.sidebar.multiselect("Select Offense Group(s)", crime_types, default=crime_types[:5])

    if len(years) > 0:
        year_filter = st.sidebar.slider("Select Year Range", min_value=min(years), max_value=max(years),
                                        value=(min(years), max(years)))
    else:
        year_filter = (0, 0)

    filtered_df = df[
        (df["OffenseGroup"].isin(crime_filter)) &
        (df["Year"].between(year_filter[0], year_filter[1]))
    ].copy()

    st.title("📊 Crime Pattern Detection & Prediction Dashboard")
    st.markdown("Explore spatio-temporal crime patterns for **Greater Manchester Police**.")

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        total_crimes = int(df["CrimeCount"].sum())
        st.metric("Total Crimes (all data)", f"{total_crimes:,}")
    with col2:
        filtered_crimes = int(filtered_df["CrimeCount"].sum()) if not filtered_df.empty else 0
        st.metric("Filtered Crimes", f"{filtered_crimes:,}")
    with col3:
        if not filtered_df.empty:
            try:
                top_offense = filtered_df.groupby("OffenseGroup")["CrimeCount"].sum().idxmax()
            except Exception:
                top_offense = filtered_df["OffenseGroup"].mode()[0]
        else:
            top_offense = "N/A"
        st.metric("Top Offense", top_offense)

    st.markdown("---")

    # Crime Count by Offense Group (bar chart) with comma formatting
    st.subheader("📌 Crime Count by Offense Group")
    if not filtered_df.empty:
        offense_dist = (
            filtered_df.groupby("OffenseGroup")["CrimeCount"]
            .sum()
            .reset_index()
            .sort_values("CrimeCount", ascending=False)
        )
    else:
        offense_dist = pd.DataFrame(columns=["OffenseGroup", "CrimeCount"])

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    if not offense_dist.empty:
        sns.barplot(x="CrimeCount", y="OffenseGroup", data=offense_dist, palette="Blues_d", ax=ax1)
        ax1.set_title("Distribution of Crimes by Offense Group", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Crimes")
        ax1.set_ylabel("")
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    else:
        ax1.text(0.5, 0.5, "No data for selected filters", ha="center", va="center")
        ax1.set_axis_off()
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.markdown("---")

    # Yearly trend with comma formatting
    st.subheader("📈 Yearly Crime Trend")
    if not filtered_df.empty:
        crime_trend = filtered_df.groupby("Year")["CrimeCount"].sum().reset_index().sort_values("Year")
    else:
        crime_trend = pd.DataFrame(columns=["Year", "CrimeCount"])

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    if not crime_trend.empty:
        sns.lineplot(x="Year", y="CrimeCount", data=crime_trend, marker="o", ax=ax2)
        ax2.set_title("Yearly Trend of Crimes", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Number of Crimes")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    else:
        ax2.text(0.5, 0.5, "No data for selected filters", ha="center", va="center")
        ax2.set_axis_off()
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown("---")

    # Greater Manchester map
    st.subheader("🗺️ Greater Manchester Police Boundary")
    m = folium.Map(location=[53.4808, -2.2426], zoom_start=9, tiles="cartodbpositron")
    folium.GeoJson(
        gm_map.to_json(),
        name="Greater Manchester Police",
        style_function=lambda x: {"color": "red", "weight": 2, "fillOpacity": 0.1}
    ).add_to(m)
    st_map = st_folium(m, width=600, height=350)

    st.markdown("---")

    # Offense co-occurrence network from filtered_df
    st.subheader("🔗 Offense Co-occurrence Network")
    G = nx.Graph()
    if not filtered_df.empty:
        cooccurrence = filtered_df.groupby(["Year", "PoliceForce"])["OffenseGroup"].apply(lambda x: list(set(x))).reset_index(drop=True)
        edges = []
        for groups in cooccurrence:
            if not groups or len(groups) < 2:
                continue
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    edges.append((groups[i], groups[j]))
        if edges:
            G.add_edges_from(edges)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=500, ax=ax3)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax3)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax3)
        ax3.set_title("Offense Group Co-occurrence Network", fontsize=11, fontweight="bold")
        ax3.set_axis_off()
    else:
        ax3.text(0.5, 0.5, "No co-occurrence data for selected filters", ha="center", va="center")
        ax3.set_axis_off()
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown("---")

    # Crime forecasting (filtered and restricted to Greater Manchester)
    st.subheader("🔮 Crime Forecasting (Next 5 Years)")
    gm_crime = filtered_df[filtered_df['PoliceForce'].str.contains("Greater Manchester", case=False, na=False)]
    forecast = None
    model = None
    crime_forecast = pd.DataFrame()
    if not gm_crime.empty:
        crime_forecast = gm_crime.groupby("Year")["CrimeCount"].sum().reset_index().sort_values("Year")
        crime_forecast = crime_forecast.rename(columns={"Year": "ds", "CrimeCount": "y"})
        crime_forecast["ds"] = pd.to_datetime(crime_forecast["ds"].astype(str), format="%Y")

        model = Prophet(yearly_seasonality=True)
        try:
            model.fit(crime_forecast)
            future = model.make_future_dataframe(periods=5, freq="Y")
            forecast = model.predict(future)

            fig4, ax4 = plt.subplots(figsize=(8, 3))
            sns.lineplot(x="ds", y="y", data=crime_forecast, marker="o", label="Historical", ax=ax4)
            sns.lineplot(x="ds", y="yhat", data=forecast, label="Forecast", ax=ax4)
            ax4.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.25)
            ax4.set_title("Crime Forecasting (Prophet) - Historical vs Forecast", fontsize=11, fontweight="bold")
            ax4.set_ylabel("Number of Crimes")
            ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

            fig5 = model.plot_components(forecast)
            # Prophet returns a matplotlib figure; format y-axes on each axis for readability
            for ax in fig5.get_axes():
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}" if abs(x) >= 1 else f"{x:.0f}"))
            plt.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)
        except Exception as e:
            st.warning(f"Forecasting failed: {e}")
    else:
        st.info("No Greater Manchester crime rows in the filtered data to forecast.")

with tab_export:
    st.title("📄 Export Report")
    st.markdown("Generate and download a PDF report that includes KPIs and all charts shown on the dashboard.")

    if st.button("Generate PDF Report"):
        saved_images = []

        # Recreate & save offense distribution with formatted axis
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            if not offense_dist.empty:
                sns.barplot(x="CrimeCount", y="OffenseGroup", data=offense_dist, palette="Blues_d", ax=ax)
                ax.set_title("Distribution of Crimes by Offense Group", fontsize=12, fontweight="bold")
                ax.set_xlabel("Number of Crimes")
                ax.set_ylabel("")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
            else:
                ax.text(0.5, 0.5, "No data for selected filters", ha="center", va="center")
                ax.set_axis_off()
            fig.savefig("offense_distribution.png", bbox_inches="tight")
            saved_images.append("offense_distribution.png")
            plt.close(fig)
        except Exception:
            pass

        # Recreate & save yearly trend with formatted axis
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            if not crime_trend.empty:
                sns.lineplot(x="Year", y="CrimeCount", data=crime_trend, marker="o", ax=ax)
                ax.set_title("Yearly Trend of Crimes", fontsize=11, fontweight="bold")
                ax.set_ylabel("Number of Crimes")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
            else:
                ax.text(0.5, 0.5, "No data for selected filters", ha="center", va="center")
                ax.set_axis_off()
            fig.savefig("yearly_trend.png", bbox_inches="tight")
            saved_images.append("yearly_trend.png")
            plt.close(fig)
        except Exception:
            pass

        # Recreate & save offense network
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=0.5, seed=42)
                nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=500, ax=ax)
                nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
                ax.set_title("Offense Group Co-occurrence Network", fontsize=11, fontweight="bold")
                ax.set_axis_off()
            else:
                ax.text(0.5, 0.5, "No co-occurrence data for selected filters", ha="center", va="center")
                ax.set_axis_off()
            fig.savefig("offense_network.png", bbox_inches="tight")
            saved_images.append("offense_network.png")
            plt.close(fig)
        except Exception:
            pass

        # Recreate & save forecast plot (if forecast exists)
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            if forecast is not None and not crime_forecast.empty:
                sns.lineplot(x="ds", y="y", data=crime_forecast, marker="o", label="Historical", ax=ax)
                sns.lineplot(x="ds", y="yhat", data=forecast, label="Forecast", ax=ax)
                ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.25)
                ax.set_title("Crime Forecasting (Prophet) - Historical vs Forecast", fontsize=11, fontweight="bold")
                ax.set_ylabel("Number of Crimes")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
            else:
                ax.text(0.5, 0.5, "No forecasting data for selected filters", ha="center", va="center")
                ax.set_axis_off()
            fig.savefig("forecast.png", bbox_inches="tight")
            saved_images.append("forecast.png")
            plt.close(fig)
        except Exception:
            pass

        # Save Prophet components if available and format their y-axes
        try:
            if forecast is not None and model is not None:
                comp_fig = model.plot_components(forecast)
                for ax in comp_fig.get_axes():
                    # try to format y-axis values to comma style for larger numbers
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}" if abs(x) >= 1 else f"{x:.0f}"))
                comp_fig.savefig("forecast_components.png", bbox_inches="tight")
                saved_images.append("forecast_components.png")
                plt.close(comp_fig)
        except Exception:
            pass

        # Build PDF
        pdf = FPDF(unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=14, style="B")
        pdf.cell(0, 10, "Crime Pattern Report - Greater Manchester", ln=True, align="C")
        pdf.ln(6)

        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, f"Total Crimes (all data): {total_crimes:,}")
        pdf.multi_cell(0, 6, f"Filtered Crimes: {filtered_crimes:,}")
        pdf.multi_cell(0, 6, f"Top Offense: {top_offense}")
        pdf.ln(4)
        pdf.multi_cell(0, 6, "This report summarises crime trends, offence distribution, network analysis, and forecasting for Greater Manchester Police.")
        pdf.ln(6)

        def add_figure_to_pdf(title, image_path):
            if os.path.exists(image_path):
                pdf.set_font("Arial", size=12, style="B")
                pdf.cell(0, 6, title, ln=True)
                pdf.ln(1)
                try:
                    pdf.image(image_path, w=180)
                    pdf.ln(6)
                except RuntimeError:
                    pdf.multi_cell(0, 6, f"(Could not embed image: {image_path})")
                    pdf.ln(4)

        # Insert saved images in the PDF
        add_figure_to_pdf("Crime Count by Offense Group", "offense_distribution.png")
        add_figure_to_pdf("Yearly Crime Trend", "yearly_trend.png")
        add_figure_to_pdf("Offense Co-occurrence Network", "offense_network.png")
        add_figure_to_pdf("Crime Forecast (Historical vs Forecast)", "forecast.png")
        add_figure_to_pdf("Forecast Components", "forecast_components.png")

        out_pdf = "crime_report.pdf"
        pdf.output(out_pdf)

        # Cleanup temporary image files
        for fpath in saved_images:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception:
                pass

        # Provide download
        if os.path.exists(out_pdf):
            with open(out_pdf, "rb") as f:
                pdf_bytes = f.read()
            st.success("✅ PDF report generated (crime_report.pdf)")
            st.download_button(label="Download PDF", data=pdf_bytes, file_name=out_pdf, mime="application/pdf")
        else:
            st.error("Failed to create PDF report.")
