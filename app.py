import streamlit as st
import cohere
import pandas as pd
from datetime import datetime

# 设置页面标题和基本信息
st.set_page_config(page_title="Cohere Rerank Tester", layout="wide")
st.title("Cohere Rerank Tester")

# 添加说明文字
st.markdown("""
### 使用说明
1. 输入您的 Cohere API Key（可以从 [Cohere Dashboard](https://dashboard.cohere.ai/api-keys) 获取）
2. 上传Excel文件（需包含 'query' 和 'documents' 两列，documents 列中的多个文档用换行符分隔）
3. 点击"开始Rerank"按钮进行处理
""")

# Cohere API设置
api_key = st.text_input("请输入您的 Cohere API Key:", type="password", help="在此输入您的Cohere API Key")

# 文件上传
uploaded_file = st.file_uploader("上传Excel文件", type=['xlsx', 'xls'], help="上传包含query和documents列的Excel文件")

if uploaded_file is not None and api_key:
    try:
        # 初始化Cohere客户端
        co = cohere.Client(api_key)
        
        # 读取Excel文件
        df = pd.read_excel(uploaded_file)
        st.write("数据预览：")
        st.dataframe(df.head())
        
        if st.button("开始Rerank"):
            # 验证数据格式
            if 'query' not in df.columns or 'documents' not in df.columns:
                st.error("Excel文件必须包含 'query' 和 'documents' 两列！")
            else:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    # 更新进度
                    progress = (idx + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"处理第 {idx+1}/{len(df)} 条数据...")
                    
                    # 处理documents列，分割成列表
                    documents = row['documents'].split('\n') if isinstance(row['documents'], str) else [str(row['documents'])]
                    
                    try:
                        # 调用Rerank API
                        rerank_results = co.rerank(
                            query=str(row['query']),
                            documents=documents,
                            top_n=len(documents),
                            model='rerank-english-v2.0'
                        )
                        
                        # 处理结果
                        for result in rerank_results:
                            results.append({
                                'query': row['query'],
                                'document': result.document,
                                'relevance_score': result.relevance_score,
                                'index': result.index
                            })
                    except Exception as e:
                        st.warning(f"处理第 {idx+1} 条数据时出错: {str(e)}")
                        continue
                
                status_text.text("处理完成！")
                
                if results:
                    # 显示结果
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('relevance_score', ascending=False)
                    
                    st.write("### Rerank 结果：")
                    st.dataframe(results_df)
                    
                    # 导出按钮
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="导出结果为CSV",
                        data=csv,
                        file_name=f'rerank_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
                else:
                    st.error("处理过程中没有产生有效结果。")
                
    except Exception as e:
        st.error(f"处理过程中出现错误: {str(e)}")
elif uploaded_file is not None and not api_key:
    st.warning("请先输入您的 Cohere API Key") 
