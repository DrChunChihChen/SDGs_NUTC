import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

# Set page configuration
st.set_page_config(
    page_title="國立臺中科技大學 SDGs 課程統計表",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #333333;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 40px;
        font-weight: bold;
    }
    .metric-title {
        font-size: 16px;
        color: #666666;
    }
    .blue-value {
        color: #3366cc;
    }
    .green-value {
        color: #2e8b57;
    }
    .orange-value {
        color: #ff8c00;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 2px solid #4169e1;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">國立臺中科技大學 SDGs 課程統計表</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("上傳您的 Excel 課程資料", type=["xlsx", "xls"])


# Function to extract SDG numbers from the 'final SDGs' column
def extract_sdg_numbers(sdg_string):
    if pd.isna(sdg_string):
        return []

    # Extract numbers after "SDG" using regex
    matches = re.findall(r'SDG(\d+)', str(sdg_string))
    return [int(match) for match in matches]


# Function to process Excel file and create dashboard
def create_dashboard(df):
    # Extract the academic year from the data (開課年度)
    if '開課年度' in df.columns:
        academic_year = df['開課年度'].iloc[0] if not df.empty else "未知"
        title = f"{academic_year} 學年國立臺中科技大學 SDGs 課程統計表"
    else:
        title = "國立臺中科技大學 SDGs 課程統計表"
        academic_year = "未知"

    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)

    # Right aligned "實際數據展示版" button
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("""
            <div style="text-align: right;">
                <button style="
                    background-color: white;
                    color: green;
                    border: 1px solid green;
                    border-radius: 20px;
                    padding: 5px 15px;
                    cursor: pointer;">
                    實際數據展示版
                </button>
            </div>
        """, unsafe_allow_html=True)

    # Create a new column with SDG numbers as a list
    df['SDG_Numbers'] = df['final SDGs'].apply(extract_sdg_numbers)

    # Count courses
    total_courses = len(df)

    # Count SDG-related courses (those with at least one SDG)
    sdg_related_courses = df[df['SDG_Numbers'].apply(len) > 0].shape[0]

    # Calculate percentage
    sdg_percentage = (sdg_related_courses / total_courses * 100) if total_courses > 0 else 0

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">總課程數 ({academic_year}學年)</div>
            <div class="metric-value blue-value">{total_courses}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">SDG 相關課程數</div>
            <div class="metric-value green-value">{sdg_related_courses}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">SDG 課程佔比</div>
            <div class="metric-value orange-value">{sdg_percentage:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Create tabs for different visualizations
    tabs = st.tabs(["SDG 分佈圖", "科系 SDG 分佈表", "SDG 課程統計", "課程資料探索"])

    # SDG Distribution Tab
    with tabs[0]:
        # Count courses by SDG
        sdg_counts = {i: 0 for i in range(1, 18)}
        for _, row in df.iterrows():
            for sdg_num in row['SDG_Numbers']:
                if 1 <= sdg_num <= 17:  # Ensure valid SDG number
                    sdg_counts[sdg_num] += 1

        # Create DataFrame for the bar chart
        sdg_labels = [
            'SDG1: 消除貧窮', 'SDG2: 消除飢餓', 'SDG3: 良好健康與福祉', 'SDG4: 優質教育',
            'SDG5: 性別平等', 'SDG6: 清潔飲水和衛生設施', 'SDG7: 可負擔能源',
            'SDG8: 體面工作和經濟增長', 'SDG9: 工業, 創新和基礎設施', 'SDG10: 減少不平等',
            'SDG11: 永續城市', 'SDG12: 負責任消費與生產', 'SDG13: 氣候行動',
            'SDG14: 海洋生態', 'SDG15: 陸地生態', 'SDG16: 和平與正義制度', 'SDG17: 全球夥伴關係'
        ]

        df_sdg_distribution = pd.DataFrame({
            'SDG': sdg_labels,
            'Count': [sdg_counts[i] for i in range(1, 18)]
        })

        # Create horizontal bar chart
        st.markdown("### 各項 SDG 課程分佈")
        fig_bar = px.bar(
            df_sdg_distribution,
            x='Count',
            y='SDG',
            orientation='h',
            color='Count',
            color_continuous_scale='RdBu',
            height=600
        )

        fig_bar.update_layout(
            xaxis_title="課程數量",
            yaxis_title="",
            yaxis={'categoryorder': 'total descending'},
            margin=dict(l=0, r=10, t=10, b=0),
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # Department vs SDG Heatmap Tab
    with tabs[1]:
        if '科系' in df.columns:
            # Create a cross-tabulation of departments and SDGs
            dept_sdg_data = {}

            # Initialize with zero counts
            departments = df['科系'].unique()
            for dept in departments:
                dept_sdg_data[dept] = {i: 0 for i in range(1, 18)}

            # Count SDGs by department
            for _, row in df.iterrows():
                dept = row['科系']
                for sdg_num in row['SDG_Numbers']:
                    if 1 <= sdg_num <= 17:  # Ensure valid SDG number
                        dept_sdg_data[dept][sdg_num] += 1

            # Create DataFrame for heatmap
            dept_sdg_rows = []
            for dept, sdg_counts in dept_sdg_data.items():
                row = {'科系': dept}
                for i in range(1, 18):
                    row[f'SDG{i}'] = sdg_counts[i]
                row['總計'] = sum(sdg_counts.values())
                dept_sdg_rows.append(row)

            df_heatmap = pd.DataFrame(dept_sdg_rows)
            df_heatmap = df_heatmap.sort_values('總計', ascending=False)

            # Add filter for top departments
            top_n = st.slider("顯示前幾名科系", min_value=5, max_value=len(df_heatmap), value=min(20, len(df_heatmap)),
                              step=1)
            df_heatmap_filtered = df_heatmap.head(top_n)

            # Display the table with totals
            st.markdown("### 科系 SDG 課程數據表格")
            # Make the table sortable by clicking on column headers
            st.dataframe(df_heatmap_filtered.style.highlight_max(axis=0, subset=[f'SDG{i}' for i in range(1, 18)]),
                         height=600)
        else:
            st.error("找不到科系欄位，無法建立科系 SDG 分佈表")

    # SDG Course Statistics Tab
    with tabs[2]:
        st.markdown("### SDG 課程統計")

        # Create a dataframe to show the frequency and percentage of each SDG
        sdg_counts = {i: 0 for i in range(1, 18)}
        for _, row in df.iterrows():
            for sdg_num in row['SDG_Numbers']:
                if 1 <= sdg_num <= 17:  # Ensure valid SDG number
                    sdg_counts[sdg_num] += 1

        # SDG name mappings
        sdg_names = {
            1: "消除貧窮",
            2: "消除飢餓",
            3: "良好健康與福祉",
            4: "優質教育",
            5: "性別平等",
            6: "清潔飲水和衛生設施",
            7: "可負擔能源",
            8: "體面工作和經濟增長",
            9: "工業, 創新和基礎設施",
            10: "減少不平等",
            11: "永續城市",
            12: "負責任消費與生產",
            13: "氣候行動",
            14: "海洋生態",
            15: "陸地生態",
            16: "和平與正義制度",
            17: "全球夥伴關係"
        }

        # Create DataFrame for statistics
        stats_data = []
        for i in range(1, 18):
            stats_data.append({
                'SDG編號': f'SDG{i}',
                'SDG名稱': sdg_names[i],
                '課程數量': sdg_counts[i],
                '佔總SDG課程比例(%)': round(sdg_counts[i] / sum(sdg_counts.values()) * 100, 2) if sum(
                    sdg_counts.values()) > 0 else 0
            })

        df_stats = pd.DataFrame(stats_data)

        # Display statistics in a table
        st.dataframe(df_stats, height=600)

        # Create distribution visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### SDG 課程數量分佈")

            # Bar chart for SDG course counts
            fig_bar_stats = px.bar(
                df_stats,
                x='SDG編號',
                y='課程數量',
                color='課程數量',
                labels={'課程數量': '課程數量', 'SDG編號': 'SDG編號'},
                color_continuous_scale='Viridis',
                height=400
            )

            fig_bar_stats.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar_stats, use_container_width=True)

        with col2:
            st.markdown("### SDG 課程佔比分佈")

            # Pie chart for SDG distribution
            fig_pie_stats = px.pie(
                df_stats,
                values='課程數量',
                names='SDG編號',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )

            fig_pie_stats.update_layout(height=400)
            st.plotly_chart(fig_pie_stats, use_container_width=True)

        # Display courses with multiple SDGs
        st.markdown("### 多重 SDG 課程分析")

        # Count number of SDGs per course
        df['SDG_Count'] = df['SDG_Numbers'].apply(len)

        # Create histogram of SDG counts per course
        fig_hist = px.histogram(
            df,
            x='SDG_Count',
            nbins=17,
            labels={'SDG_Count': 'SDG 數量', 'count': '課程數量'},
            title='每個課程的 SDG 數量分佈',
            color_discrete_sequence=['#3366cc']
        )

        fig_hist.update_layout(
            xaxis=dict(tickmode='linear', dtick=1),
            bargap=0.2
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    # Data Exploration Tab
    with tabs[3]:
        st.markdown("### 課程資料探索")

        # Create options for filtering
        filter_cols = st.columns(3)

        # Filter by department if available
        if '科系' in df.columns:
            with filter_cols[0]:
                all_depts = sorted(df['科系'].unique())
                selected_dept = st.multiselect(
                    "選擇科系",
                    options=["全部"] + all_depts,
                    default=["全部"]
                )

                # Handle "全部" selection
                if "全部" in selected_dept:
                    selected_dept = all_depts

        # Filter by SDG
        with filter_cols[1]:
            all_sdgs = list(range(1, 18))
            selected_sdgs = st.multiselect(
                "選擇 SDG",
                options=["全部"] + [f"SDG{i}" for i in all_sdgs],
                default=["全部"]
            )

            # Convert SDG strings to numbers and handle "全部" selection
            if "全部" in selected_sdgs:
                selected_sdgs = all_sdgs
            else:
                selected_sdgs = [int(sdg.replace("SDG", "")) for sdg in selected_sdgs]

        # Filter by course year if available
        if '開課年度' in df.columns:
            with filter_cols[2]:
                all_years = sorted(df['開課年度'].unique())
                selected_years = st.multiselect(
                    "選擇學年",
                    options=["全部"] + [str(year) for year in all_years],
                    default=["全部"]
                )

                # Handle "全部" selection
                if "全部" in selected_years:
                    selected_years = all_years
                else:
                    selected_years = [int(year) if year.isdigit() else year for year in selected_years]

        # Apply filters
        filtered_df = df.copy()

        if '科系' in df.columns and selected_dept:
            filtered_df = filtered_df[filtered_df['科系'].isin(selected_dept)]

        if selected_sdgs:
            filtered_df = filtered_df[filtered_df['SDG_Numbers'].apply(
                lambda x: any(sdg in x for sdg in selected_sdgs)
            )]

        if '開課年度' in df.columns and selected_years:
            filtered_df = filtered_df[filtered_df['開課年度'].isin(selected_years)]

        # Display filtered data
        selected_columns = st.multiselect(
            "選擇顯示欄位",
            options=df.columns.tolist(),
            default=['課程名稱', '科系', '開課老師', '學分數', 'final SDGs'] if all(
                col in df.columns for col in ['課程名稱', '科系', '開課老師', '學分數', 'final SDGs']) else df.columns[
                                                                                                            :5].tolist()
        )

        if not selected_columns:
            st.warning("請選擇至少一個欄位以顯示資料")
        else:
            st.dataframe(filtered_df[selected_columns], height=600)

            # Download button for filtered data
            csv = filtered_df[selected_columns].to_csv(index=False)
            st.download_button(
                label="下載篩選後的資料 (CSV)",
                data=csv,
                file_name="filtered_sdg_courses.csv",
                mime="text/csv",
            )

        # Display course count by SDG for filtered data
        if not filtered_df.empty:
            st.markdown("### 篩選後的 SDG 課程分佈")

            # Count courses by SDG in filtered data
            filtered_sdg_counts = {i: 0 for i in range(1, 18)}
            for _, row in filtered_df.iterrows():
                for sdg_num in row['SDG_Numbers']:
                    if 1 <= sdg_num <= 17:
                        filtered_sdg_counts[sdg_num] += 1

            # Create DataFrame for the chart and tables
            filtered_df_sdg = pd.DataFrame({
                'SDG': [f'SDG{i}' for i in range(1, 18)],
                'Count': [filtered_sdg_counts[i] for i in range(1, 18)]
            })

            # Bar chart instead of table for Sum of each SDG after selection
            st.markdown(f"#### 各 SDG 課程數量 (SDG課程總數: {filtered_df_sdg['Count'].sum()})")

            # Create bar chart for SDG counts
            fig_bar = px.bar(
                filtered_df_sdg,
                x='SDG',
                y='Count',
                color='Count',
                color_continuous_scale='Viridis',
                labels={'Count': '課程數量', 'SDG': 'SDG編號'}
            )

            fig_bar.update_layout(
                xaxis_tickangle=-45,
                height=400,
                yaxis_title="課程數量",
                xaxis_title="SDG編號"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

            # Table 2: Sum of total SDGs after selection
            total_sdg_courses = filtered_df_sdg['Count'].sum()
            total_courses = len(filtered_df)

            # Create a DataFrame for the total summary
            total_summary = pd.DataFrame({
                '項目': ['SDG 課程總數', '篩選後課程總數', 'SDG 課程佔比'],
                '數值': [
                    f"{total_sdg_courses}",
                    f"{total_courses}",
                    f"{(total_sdg_courses / total_courses * 100):.1f}%" if total_courses > 0 else "0.0%"
                ]
            })


            st.markdown("#### SDG 課程分佈圖")
            fig_pie = px.pie(
                filtered_df_sdg[filtered_df_sdg['Count'] > 0],
                values='Count',
                names='SDG',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=500
            )

            st.plotly_chart(fig_pie, use_container_width=True)

# If a file is uploaded, process it
if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)

        # Check if the required column exists
        if 'final SDGs' not in df.columns:
            st.error("上傳的 Excel 文件必須包含 'final SDGs' 欄位。請確認您的文件格式正確。")
        else:
            # Create the dashboard
            create_dashboard(df)
    except Exception as e:
        st.error(f"處理文件時發生錯誤: {e}")
else:
    # Display instructions when no file is uploaded
    st.info("""
    請上傳包含以下欄位的 Excel 文件:

    建立時間, 開課年度, 開課學期, 名稱, 級別, 名校名, 課程名稱, 科系, 科系英文名稱, 學程, 年級, 開課老師, 學分數,
    課程大綱, 課程連結, 備註, 男學生修課人數, 女學生修課人數, 其他學生修課人數, 部別_學制, 學程_部, 校訂_半全年,
    科目類別, 教學型態, 必選修, 全英語教學, 跨校選修, 系科屬性, 總修課人數, 授課語言, 課號, final SDGs

    其中 'final SDGs' 欄位是必須的，它應該包含格式如 "SDG1, SDG3, SDG4" 的數據。
    """)

    # Create a sample dataframe for demonstration
    st.markdown("### 範例數據")
    sample_data = {
        '建立時間': ['2023-09-01', '2023-09-01', '2023-09-01'],
        '開課年度': [113, 113, 113],
        '課程名稱': ['永續發展導論', '環境科學概論', '企業社會責任'],
        '科系': ['通識教育中心', '環境工程系', '企業管理系'],
        '開課老師': ['王老師', '李老師', '張老師'],
        '學分數': [2, 3, 3],
        'final SDGs': ['SDG4, SDG17', 'SDG6, SDG13, SDG14, SDG15', 'SDG8, SDG12']
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)

    # Convert sample data to csv for download
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="下載範例數據 (CSV)",
        data=csv,
        file_name="sample_sdg_courses.csv",
        mime="text/csv",
    )
