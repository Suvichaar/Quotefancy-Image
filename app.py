import streamlit as st
import pandas as pd
import json
import re
import time
from io import StringIO

st.set_page_config(page_title="📸 Image Batch Tool", layout="wide")

# ===================== 🗂️ Tab Setup =====================
tab1, tab2 = st.tabs(["📤 Upload & Process CSV", "🔍 ALT Text Preview"])

# ===================== 📤 TAB 1: Upload & Process =====================
with tab1:
    st.title("📤 Image ALT Text Generator & JSONL Preparer")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Clean the data
        df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        df.replace("NA", pd.NA, inplace=True)

        required_cols = ['Keyword', 'Filename', 'CDN_URL']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            df.dropna(subset=required_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Generate custom_id
            custom_ids = []
            for idx, row in df.iterrows():
                filename = row['Filename'].strip()
                filename_no_ext = re.sub(r"\\.[^.]+$", "", filename)
                custom_ids.append(f"{idx + 1}-{filename_no_ext}")
            df.insert(0, "custom_id", custom_ids)

            # Generate prompt
            df["prompt"] = df["Keyword"].apply(
                lambda keyword: f"Given the following image URL of a famous personality, generate a short ALT text (max 1–2 sentences) that introduces the {keyword}, including their name, legacy, or profession in a respectful tone suitable for accessibility or SEO purposes."
            )

            # Show processed data
            st.subheader("✅ Preview Processed Data")
            st.dataframe(df.head())

            # Save CSV
            timestamp = str(int(time.time()))
            csv_output_filename = f"Image_Data_Custom_id_prompt_{timestamp}.csv"
            df.to_csv(csv_output_filename, index=False)
            st.download_button("📥 Download Processed CSV", data=df.to_csv(index=False), file_name=csv_output_filename, mime="text/csv")

            # Generate JSONL
            json_records = []
            for _, row in df.iterrows():
                record = {
                    "custom_id": row["custom_id"],
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": "gpt-4o-global-batch",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful and professional assistant with expertise in creating descriptive ALT texts that are accessible, informative, and optimized for SEO. Respond with clarity and respect."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": row["prompt"]
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": row["CDN_URL"],
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 1000
                    }
                }
                json_records.append(record)

            jsonl_str = '\n'.join(json.dumps(record) for record in json_records)
            jsonl_output_filename = f"azure_image_batch_requests_{timestamp}.jsonl"
            st.download_button("📥 Download JSONL File", data=jsonl_str, file_name=jsonl_output_filename, mime="application/jsonl")

            # Store for next tab
            st.session_state["processed_df"] = df

# ===================== 🔍 TAB 2: ALT Text Preview =====================
with tab2:
    st.title("🔍 ALT Text Prompt Preview")
    if "processed_df" in st.session_state:
        st.dataframe(st.session_state["processed_df"][["custom_id", "Keyword", "CDN_URL", "prompt"]])
    else:
        st.info("Please upload and process a CSV in Tab 1 first.")
