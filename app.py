import streamlit as st
import pandas as pd
import json
import re
import time
import unicodedata
from io import StringIO
from collections import defaultdict
import base64
import random
import string
from datetime import datetime, timezone
import ftfy
from openai import AzureOpenAI
from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
    ContentSettings
)

st.set_page_config(page_title="📸 Image Batch Tool", layout="wide")

# ===================== 🗂️ Tab Setup =====================
tabs = st.tabs([
    "🖼️ Azure OpenAI ALT Text Generator for Image Metadata",
    "🧠 Azure OpenAI Batch Result Retriever + Blob Uploader",
    "📥 Map ALT Text from JSONL",
    "🧹 Clean Final ALT Text File",
    "🔄 Distribute ALT Images to Author Rows",
    "🧱 Generate CDN Resize URLs",
    "🔁 Fill Missing Image & ALT Text",
    "🧩 Add Suvichaar Metadata",
    "🪄 Wrap Columns with {{...}}",
    "🎬 Add Random Video Rows + Circular Navigation",
    "🧾 Final Column Reorder"
])

(tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11) = tabs


# ===================== 📤 TAB 1: Upload & Process =====================
with tab1:
    
    # ============================ 🔐 Azure Configuration ============================
    client = AzureOpenAI(
        api_key=st.secrets["azure_api_key"],
        api_version="2025-03-01-preview",
        azure_endpoint=st.secrets["azure_endpoint"]
    )
    
    deployment_model = "gpt-4o-global-batch"
    
    # ============================ 🎯 App UI ============================
    st.title("🖼️ Azure OpenAI ALT Text Generator for Image Metadata")
    
    uploaded_file = st.file_uploader("📤 Upload CSV with 'Keyword', 'Filename', and 'CDN_URL' columns", type="csv")
    
    if uploaded_file:
        ts = str(int(time.time()))
        df = pd.read_csv(uploaded_file)
    
        # ============================ 🧹 Clean CSV ============================
        df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        df.replace("NA", pd.NA, inplace=True)
        required_cols = ['Keyword', 'Filename', 'CDN_URL']
    
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ CSV must contain the columns: {required_cols}")
        else:
            df.dropna(subset=required_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)
    
            # ============================ 🆔 Generate custom_id and prompt ============================
            custom_ids = []
            for idx, row in df.iterrows():
                filename = row['Filename'].strip()
                filename_no_ext = re.sub(r"\.[^.]+$", "", filename)
                custom_ids.append(f"{idx + 1}-{filename_no_ext}")
    
            df.insert(0, "custom_id", custom_ids)
            df["prompt"] = df["Keyword"].apply(
                lambda keyword: f"Given the following image URL of a famous personality, generate a short ALT text (max 1–2 sentences) that introduces the {keyword}, including their name, legacy, or profession in a respectful tone suitable for accessibility or SEO purposes."
            )
    
            # ============================ 💾 Show and Download CSV ============================
            csv_output_filename = f"Image_Data_Custom_id_prompt_{ts}.csv"
            st.success("✅ Data processed. Preview below:")
            st.dataframe(df.head())
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download CSV with Prompts", csv_buffer.getvalue(), file_name=csv_output_filename, mime="text/csv")
    
            # ============================ 🔄 Generate JSONL ============================
            st.subheader("🧠 Preparing Azure JSONL for batch inference...")
    
            json_records = []
            for _, row in df.iterrows():
                json_records.append({
                    "custom_id": row["custom_id"],
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": deployment_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful and professional assistant with expertise in creating descriptive ALT texts that are accessible, informative, and optimized for SEO. Respond with clarity and respect."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": row["prompt"]},
                                    {"type": "image_url", "image_url": {"url": row["CDN_URL"], "detail": "high"}}
                                ]
                            }
                        ],
                        "max_tokens": 1000
                    }
                })
    
            jsonl_filename = f"azure_image_batch_requests_{ts}.jsonl"
            jsonl_str = "\n".join([json.dumps(item) for item in json_records])
            st.download_button("📥 Download JSONL File", data=jsonl_str, file_name=jsonl_filename, mime="application/jsonl")
    
            # ============================ ⬆️ Upload JSONL to Azure ============================
            if st.button("🚀 Upload JSONL to Azure and Submit Batch Job"):
                with st.spinner("Uploading JSONL to Azure..."):
                    batch_file = client.files.create(
                        file=StringIO(jsonl_str),
                        purpose="batch",
                        extra_body={"expires_after": {"seconds": 1209600, "anchor": "created_at"}}
                    )
                    file_id = batch_file.id
                    st.success("✅ JSONL uploaded to Azure.")
    
                # ============================ 🚀 Submit Batch Job ============================
                with st.spinner("Submitting batch job..."):
                    batch_job = client.batches.create(
                        input_file_id=file_id,
                        endpoint="/chat/completions",
                        completion_window="24h",
                        extra_body={"output_expires_after": {"seconds": 1209600, "anchor": "created_at"}}
                    )
                    batch_id = batch_job.id
                    st.success(f"🚀 Batch job submitted! Batch ID: {batch_id}")
    
                # ============================ 💾 Save Tracking Info ============================
                tracking_info = {
                    "ts": ts,
                    "batch_id": batch_id,
                    "file_id": file_id,
                    "jsonl_file": jsonl_filename,
                    "csv_file": csv_output_filename
                }
                track_filename = f"azure_image_batch_tracking_{ts}.json"
                st.download_button("📥 Download Tracking Info", json.dumps(tracking_info, indent=2), file_name=track_filename, mime="application/json")
    
                st.balloons()

# ===================== 📤 TAB 1: Upload & Process =====================
with tab2:
    # ============================ 🔐 Secrets / Config ============================
    azure_openai_api_key = st.secrets["azure_api_key"]
    azure_openai_endpoint = st.secrets["azure_endpoint"]
    azure_blob_connection_str = st.secrets["azure_blob_connection_string"]
    azure_blob_account_name = st.secrets["azure_blob_account_name"]
    azure_blob_account_key = st.secrets["azure_blob_account_key"]
    azure_blob_container = st.secrets["azure_blob_container"]
    
    # ============================ 🚀 Setup Clients ============================
    client = AzureOpenAI(
        api_key=azure_openai_api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=azure_openai_endpoint
    )
    
    blob_service_client = BlobServiceClient.from_connection_string(azure_blob_connection_str)
    
    # ============================ 📤 File Upload ============================
    st.title("🧠 Azure OpenAI Batch Result Retriever + Blob Uploader")
    tracking_json = st.file_uploader("📤 Upload your `azure_batch_tracking_*.json` file", type="json")
    
    if tracking_json:
        track = json.load(tracking_json)
        batch_id = track["batch_id"]
        ts = track["ts"]
        output_filename = f"batch_results_{ts}.jsonl"
    
        st.success(f"✅ Tracking info loaded. Batch ID: `{batch_id}`")
    
        # ============================ 🔍 Check Status ============================
        st.subheader("📊 Batch Status")
        with st.spinner("Checking batch status..."):
            batch_job = client.batches.retrieve(batch_id)
            status = batch_job.status
    
        st.info(f"Status: `{status}`")
    
        if status != "completed":
            st.warning("⚠️ Batch not yet completed. Please retry later.")
        else:
            output_file_id = batch_job.output_file_id or batch_job.error_file_id
    
            if output_file_id:
                with st.spinner("📥 Fetching results..."):
                    file_response = client.files.content(output_file_id)
                    raw_lines = file_response.text.strip().split('\n')
    
                result_str = "\n".join(raw_lines)
    
                # Save as file in memory
                st.success("✅ Result fetched from Azure OpenAI.")
                st.download_button("📥 Download JSONL", result_str, file_name=output_filename, mime="application/json")
    
                # ============================ ☁️ Upload to Azure Blob ============================
                st.subheader("☁️ Upload to Azure Blob Storage")
                if st.button("🚀 Upload to Azure Blob"):
                    container_client = blob_service_client.get_container_client(azure_blob_container)
    
                    container_client.upload_blob(
                        name=output_filename,
                        data=result_str,
                        overwrite=True,
                        content_settings=ContentSettings(content_type="application/json")
                    )
    
                    sas_token = generate_blob_sas(
                        account_name=azure_blob_account_name,
                        container_name=azure_blob_container,
                        blob_name=output_filename,
                        account_key=azure_blob_account_key,
                        permission=BlobSasPermissions(read=True),
                        expiry=datetime.datetime.utcnow() + datetime.timedelta(days=1)
                    )
    
                    download_url = f"https://{azure_blob_account_name}.blob.core.windows.net/{azure_blob_container}/{output_filename}?{sas_token}"
                    st.success("✅ File uploaded to Azure Blob Storage.")
                    st.markdown(f"[📎 Click here to access your file]({download_url})")
            else:
                st.error("❌ No output or error file found in the batch job.")

# ===================== 📥 TAB 3: ALT Text Mapping =====================
with tab3:
    st.title("📥 Map ALT Text from JSONL Output")
    uploaded_csv = st.file_uploader("Upload Final CSV (with custom_id)", type=["csv"], key="csv2")
    uploaded_jsonl = st.file_uploader("Upload JSONL Response File", type=["jsonl"], key="jsonl2")

    def normalize_id(cid):
        if not isinstance(cid, str):
            cid = str(cid)
        cid = unicodedata.normalize("NFKD", cid).strip().lower().replace("–", "-").replace("—", "-")
        parts = cid.split('-', 1)
        return parts[1] if len(parts) > 1 else parts[0]

    if uploaded_csv and uploaded_jsonl:
        df = pd.read_csv(uploaded_csv)
        df["custom_id_normalized"] = df["custom_id"].apply(normalize_id)

        alt_text_map = {}
        for line in uploaded_jsonl:
            try:
                obj = json.loads(line.decode("utf-8"))
                raw_id = obj.get("custom_id", "")
                norm_id = normalize_id(raw_id)
                content = obj.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content.lower().startswith("alt text:"):
                    content = content[len("alt text:"):].strip()
                alt_text_map[norm_id] = content if content else "NOT MATCHED"
            except Exception as e:
                st.warning(f"⚠️ Error parsing line: {e}")

        df["alttxt"] = df["custom_id_normalized"].apply(lambda x: alt_text_map.get(x, "NOT MATCHED"))
        df.drop(columns=["custom_id_normalized"], inplace=True)

        st.subheader("✅ Mapped ALT Text Preview")
        st.dataframe(df[["custom_id", "alttxt"]])

        timestamp = int(time.time())
        final_output_name = f"Sheet_Alttxt_Notmatched_{timestamp}.csv"
        st.download_button("📥 Download Final ALT-Text CSV", data=df.to_csv(index=False), file_name=final_output_name, mime="text/csv")

# ===================== 🧹 TAB 4: Final Cleaning =====================
with tab4:
    st.title("🧹 Clean Final CSV (Filter NOT MATCHED and Remove _1/_2 Prefixes)")
    uploaded_final_csv = st.file_uploader("Upload Mapped CSV File", type=["csv"], key="cleaner")

    if uploaded_final_csv:
        df = pd.read_csv(uploaded_final_csv)

        if 'alttxt' not in df.columns:
            st.error("❌ 'alttxt' column not found in the uploaded file.")
        else:
            original_len = len(df)
            df = df[~df['alttxt'].astype(str).str.contains("NOT MATCHED", na=False)].reset_index(drop=True)

            def should_remove(row):
                keyword = str(row['Keyword']).strip()
                filename = str(row['Filename']).strip()
                return filename.startswith(f"{keyword}_1") or filename.startswith(f"{keyword}_2")

            df = df[~df.apply(should_remove, axis=1)].reset_index(drop=True)

            st.subheader("✅ Cleaned Data Preview")
            st.dataframe(df.head())

            timestamp = int(time.time())
            output_filename = f"NOT_MATCH_1_2_Removed_output_{timestamp}.csv"
            st.download_button("📥 Download Cleaned CSV", data=df.to_csv(index=False), file_name=output_filename, mime="text/csv")

# ===================== 🧹 TAB 5: Distribute ALT Image-Text Pairs to Author Rows =====================

with tab5:
    st.title("🔄 Distribute ALT Image-Text Pairs to Author Rows")

    cleaned_csv = st.file_uploader("Upload Cleaned ALT Text CSV (with CDN_URL, alttxt, Keyword)", type=["csv"], key="dist1")
    author_data_csv = st.file_uploader("Upload Textual Data CSV (with Author)", type=["csv"], key="dist2")

    if cleaned_csv and author_data_csv:
        df1 = pd.read_csv(cleaned_csv)
        df2 = pd.read_csv(author_data_csv)

        df1['Normalized_Keyword'] = df1['Keyword'].astype(str).str.replace('-', ' ').str.replace('_', ' ').str.strip().str.lower()
        df2['Normalized_Author'] = df2['Author'].astype(str).str.strip().str.lower()

        author_image_map = defaultdict(list)
        standard_to_cdn_map = {}

        for _, row in df1.iterrows():
            key = row['Normalized_Keyword']
            standard_url = row.get('standardurl', '')
            cdn_url = row.get('CDN_URL', '')
            alt_txt = row.get('alttxt', '')

            if pd.notna(standard_url) and (standard_url, alt_txt) not in author_image_map[key]:
                author_image_map[key].append((standard_url, alt_txt))

            if pd.notna(standard_url) and pd.notna(cdn_url):
                standard_to_cdn_map[standard_url] = cdn_url

        final_rows = []
        author_row_counters = defaultdict(int)

        for _, row in df2.iterrows():
            norm_author = row['Normalized_Author']
            all_images = author_image_map.get(norm_author, [])

            start_index = author_row_counters[norm_author] * 8
            end_index = start_index + 8
            slice_images = all_images[start_index:end_index]
            author_row_counters[norm_author] += 1

            new_row = row.to_dict()
            for i in range(1, 10):
                if i - 1 < len(slice_images):
                    std_url, alt = slice_images[i - 1]
                    new_row[f's{i}image1'] = std_url
                    new_row[f's{i}alt1'] = alt
                    new_row[f's{i}cdnurl'] = standard_to_cdn_map.get(std_url, "")
                else:
                    new_row[f's{i}image1'] = ""
                    new_row[f's{i}alt1'] = ""
                    new_row[f's{i}cdnurl'] = ""

            final_rows.append(new_row)

        final_df = pd.DataFrame(final_rows)
        final_df.drop(columns=['Normalized_Author'], inplace=True, errors='ignore')

        st.subheader("✅ Final Distribution Preview")
        st.dataframe(final_df.head())

        timestamp = pd.Timestamp.now().timestamp()
        output_file = f"Distribution_Data_{int(timestamp)}.csv"
        st.download_button("📥 Download Distributed CSV", data=final_df.to_csv(index=False), file_name=output_file, mime="text/csv")

# ===================== 🧹 TAB 6: Generate Resized Image CDN URLs =====================
with tab6:
    st.title("🧱 Generate Resized Image CDN URLs")
    uploaded_resize_csv = st.file_uploader("Upload CSV with 's1cdnurl' column", type=["csv"], key="resize")

    if uploaded_resize_csv:
        df = pd.read_csv(uploaded_resize_csv)

        if 's1cdnurl' not in df.columns:
            st.error("❌ 's1cdnurl' column not found in uploaded file.")
        else:
            # Remove s2cdnurl to s9cdnurl if they exist
            for i in range(2, 10):
                col = f's{i}cdnurl'
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            resize_presets = {
                "potraightcoverresize": (640, 853),
                "landscapecoverresize": (853, 640),
                "squarecoverresize": (800, 800),
                "socialthumbnailcoverresize": (300, 300),
                "nextstoryimageresize": (315, 315)
            }

            cdn_prefix_media = "https://media.suvichaar.org/"
            transformed_df = df.copy()

            for preset_name, (width, height) in resize_presets.items():
                transformed_urls = []

                for url in transformed_df["s1cdnurl"]:
                    try:
                        if not isinstance(url, str):
                            raise ValueError("Invalid URL")

                        if "suvichaar.org/media/" in url:
                            key_path = url.split("suvichaar.org/media/")[-1]
                            normalized_url = f"{cdn_prefix_media}{key_path}"
                        else:
                            raise ValueError("Invalid CDN-style URL")

                        key_value = normalized_url.replace(cdn_prefix_media, "")
                        template = {
                            "bucket": "suvichaarapp",
                            "key": key_value,
                            "edits": {
                                "resize": {
                                    "width": width,
                                    "height": height,
                                    "fit": "cover"
                                }
                            }
                        }

                        encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
                        final_url = f"{cdn_prefix_media}{encoded}"
                        transformed_urls.append(final_url)
                    except Exception:
                        transformed_urls.append("ERROR")

                transformed_df[preset_name] = transformed_urls

            st.subheader("✅ Resized CDN URL Preview")
            st.dataframe(transformed_df.head())

            timestamp = int(time.time())
            output_filename = f"Resizer_Added_{timestamp}.csv"
            st.download_button("📥 Download Resized URL CSV", data=transformed_df.to_csv(index=False), file_name=output_filename, mime="text/csv")

# ===================== 🧹 TAB 7: Generate Resized Image CDN URLs =====================

with tab7:
    st.title("🔁 Fill Missing Image & ALT Text Fields via Rotation")
    uploaded_fill_csv = st.file_uploader("Upload the final_consistent_author_rows CSV file", type=["csv"], key="fill")

    if uploaded_fill_csv:
        df = pd.read_csv(uploaded_fill_csv)

        image_cols = [col for col in df.columns if col.startswith("s") and col.endswith("image1")]
        alt_cols = [col for col in df.columns if col.startswith("s") and col.endswith("alt1")]

        image_cols.sort()
        alt_cols.sort()

        def fill_by_rotation(row, columns):
            values = [row[col] for col in columns if pd.notna(row[col]) and row[col] != ""]
            if not values:
                return row
            i = 0
            for col in columns:
                if pd.isna(row[col]) or row[col] == "":
                    row[col] = values[i % len(values)]
                    i += 1
            return row

        df = df.apply(lambda row: fill_by_rotation(row, image_cols), axis=1)
        df = df.apply(lambda row: fill_by_rotation(row, alt_cols), axis=1)

        st.subheader("✅ Preview After Filling Missing Fields")
        st.dataframe(df.head())

        timestamp = int(pd.Timestamp.now().timestamp())
        output_filename = f"Missing_Field_Filled_{timestamp}.csv"
        st.download_button("📥 Download Completed CSV", data=df.to_csv(index=False), file_name=output_filename, mime="text/csv")

# ===================== 🧹 TAB 8:Add Suvichaar Metadata + UUID + Canonical URLs =====================

with tab8:
    st.title("🧩 Add Suvichaar Metadata + UUID + Canonical URLs")
    uploaded_meta_csv = st.file_uploader("Upload the CSV containing storytitle and content data", type=["csv"], key="meta")

    if uploaded_meta_csv:
        df = pd.read_csv(uploaded_meta_csv)

        def canurl(title):
            if not title or not isinstance(title, str):
                raise ValueError("Invalid title: Title must be a non-empty string.")
            slug = re.sub(r'[^a-z0-9-]', '', re.sub(r'\s+', '-', title.lower())).strip('-')
            alphabet = string.ascii_letters + string.digits + "_-"
            nano_id = ''.join(random.choices(alphabet, k=10)) + "_G"
            slug_nano = f"{slug}_{nano_id}"
            return [nano_id, slug_nano,
                    f"https://suvichaar.org/stories/{slug_nano}",
                    f"https://stories.suvichaar.org/{slug_nano}.html"]

        def generate_iso_time():
            now = datetime.now(timezone.utc)
            return now.strftime('%Y-%m-%dT%H:%M:%S+00:00')

        static_metadata = {
            "lang": "en-US",
            "storygeneratorname": "Suvichaar Board",
            "contenttype": "Article",
            "storygeneratorversion": "1.0.0",
            "sitename": "Suvichaar",
            "generatorplatform": "Suvichaar",
            "sitelogo96x96": "https://media.suvichaar.org/filters:resize/96x96/media/brandasset/suvichaariconblack.png",
            "sitelogo32x32": "https://media.suvichaar.org/filters:resize/32x32/media/brandasset/suvichaariconblack.png",
            "sitelogo192x192": "https://media.suvichaar.org/filters:resize/192x192/media/brandasset/suvichaariconblack.png",
            "sitelogo144x144": "https://media.suvichaar.org/filters:resize/144x144/media/brandasset/suvichaariconblack.png",
            "sitelogo92x92": "https://media.suvichaar.org/filters:resize/92x92/media/brandasset/suvichaariconblack.png",
            "sitelogo180x180": "https://media.suvichaar.org/filters:resize/180x180/media/brandasset/suvichaariconblack.png",
            "publisher": "Suvichaar",
            "publisherlogosrc": "https://media.suvichaar.org/media/brandasset/suvichaariconblack.png",
            "gtagid": "G-2D5GXVRK1E",
            "organization": "Suvichaar",
            "publisherlogoalt": "Suvichaarlogo",
            "person": "person",
            "s11btntext": "Read More",
            "s10caption1": "Your daily dose of inspiration"
        }

        user_profiles = {
            "Mayank": "https://www.instagram.com/iamkrmayank?igsh=eW82NW1qbjh4OXY2&utm_source=qr",
            "Onip": "https://www.instagram.com/onip.mathur/profilecard/?igsh=MW5zMm5qMXhybGNmdA==",
            "Naman": "https://njnaman.in/"
        }

        metadata_rows = []

        for _, row in df.iterrows():
            storytitle = str(row.get("storytitle", "")).strip()
            uuid, urlslug, canurl_val, canurl1_val = canurl(storytitle)
            published_time = generate_iso_time()
            modified_time = generate_iso_time()
            pagetitle = f"{storytitle} | Suvichaar"

            random_user = random.choice(list(user_profiles.keys()))
            random_profile = user_profiles[random_user]

            metadata = {
                "uuid": uuid,
                "urlslug": urlslug,
                "canurl": canurl_val,
                "canurl 1": canurl1_val,
                "publishedtime": published_time,
                "modifiedtime": modified_time,
                "pagetitle": pagetitle,
                "user": random_user,
                "userprofileurl": random_profile,
                **static_metadata
            }

            for key, value in metadata.items():
                row[key] = value

            metadata_rows.append(row)

        final_df = pd.DataFrame(metadata_rows)

        st.subheader("✅ Metadata Added Preview")
        st.dataframe(final_df.head())

        timestamp = int(datetime.now().timestamp())
        output_file = f"Meta_Added_{timestamp}.csv"
        st.download_button("📥 Download Metadata CSV", data=final_df.to_csv(index=False), file_name=output_file, mime="text/csv")

# ===================== 🧹 TAB 9:Wrap Column Names with {{...}} =====================

with tab9:
    st.title("🪄 Wrap Column Names with {{...}}")
    uploaded_wrap_csv = st.file_uploader("Upload the CSV file whose column headers you want to wrap with {{...}}", type=["csv"], key="wrap")

    if uploaded_wrap_csv:
        df = pd.read_csv(uploaded_wrap_csv)
        df.columns = [f"{{{{{col}}}}}" for col in df.columns]

        st.subheader("✅ Updated Column Headers Preview")
        st.dataframe(df.head())

        output_filename = f"Column_updated_{int(time.time())}.csv"
        st.download_button("📥 Download Modified CSV", data=df.to_csv(index=False), file_name=output_filename, mime="text/csv")

# ===================== 🧹 TAB 10: Add Random Video Rows + Circular Navigation (Final) =====================

with tab10:
    st.title("🎬 Add Random Video Rows + Circular Navigation (Final)")

    main_csv = st.file_uploader("📁 Upload your main dataset (quotes/stories)", type=["csv"], key="main10")
    video_csv = st.file_uploader("📁 Upload your Video-Sheets.csv file", type=["csv"], key="video10")

    if main_csv and video_csv:
        main_df = pd.read_csv(main_csv)
        video_df = pd.read_csv(video_csv)

        if "{{Author}}" in main_df.columns:
            main_df.rename(columns={"{{Author}}": "{{writername}}"}, inplace=True)
        if "{{potraightcoverresize}}" in main_df.columns:
            main_df.rename(columns={"{{potraightcoverresize}}": "{{potraightcoverurl}}"}, inplace=True)
        if "{{landscapecoverresize}}" in main_df.columns:
            main_df.rename(columns={"{{landscapecoverresize}}": "{{landscapecoverurl}}"}, inplace=True)
        if "{{squarecoverresize}}" in main_df.columns:
            main_df.rename(columns={"{{squarecoverresize}}": "{{squarecoverurl}}"}, inplace=True)
        if "{{socialthumbnailcoverresize}}" in main_df.columns:
            main_df.rename(columns={"{{socialthumbnailcoverresize}}": "{{socialthumbnailcoverurl}}"}, inplace=True)
        if "{{nextstoryimageresize}}" in main_df.columns:
            main_df.rename(columns={"{{nextstoryimageresize}}": "{{nextstorylink}}"}, inplace=True)

        selected_columns = ["{{s10video1}}", "{{hookline}}", "{{s10alt1}}", "{{videoscreenshot}}", "{{s10caption1}}"]
        random_video_rows = video_df[selected_columns].sample(n=len(main_df), replace=True).reset_index(drop=True)

        main_df["{{prevstorytitle}}"] = main_df["{{storytitle}}"].shift(1)
        main_df["{{prevstorylink}}"] = main_df["{{canurl}}"].shift(1)
        main_df.loc[0, "{{prevstorytitle}}"] = main_df.loc[main_df.index[-1], "{{storytitle}}"]
        main_df.loc[0, "{{prevstorylink}}"] = main_df.loc[main_df.index[-1], "{{canurl}}"]

        main_df["{{nextstorytitle}}"] = main_df["{{storytitle}}"].shift(-1)
        main_df["{{nextstoryimage}}"] = main_df["{{squarecoverurl}}"].shift(-1)
        main_df["{{nextstoryimagealt}}"] = main_df["{{s1alt1}}"].shift(-1)
        main_df["{{s11paragraph1}}"] = main_df["{{storytitle}}"].shift(-1)
        main_df["{{s11btnlink}}"] = main_df["{{canurl}}"].shift(-1)

        last_index = main_df.index[-1]
        main_df.loc[last_index, "{{nextstorytitle}}"] = main_df.loc[1, "{{storytitle}}"]
        main_df.loc[last_index, "{{nextstoryimage}}"] = main_df.loc[1, "{{squarecoverurl}}"]
        main_df.loc[last_index, "{{nextstoryimagealt}}"] = main_df.loc[1, "{{s1alt1}}"]
        main_df.loc[last_index, "{{s11paragraph1}}"] = main_df.loc[1, "{{storytitle}}"]
        main_df.loc[last_index, "{{s11btnlink}}"] = main_df.loc[1, "{{canurl}}"]

        final_df = pd.concat([main_df.reset_index(drop=True), random_video_rows], axis=1)

        st.subheader("✅ Final Merged Data Preview")
        st.dataframe(final_df.head())

        timestamp = int(pd.Timestamp.now().timestamp())
        output_file = f"Video_data_added_{timestamp}.csv"
        st.download_button("📥 Download Final Video CSV", data=final_df.to_csv(index=False), file_name=output_file, mime="text/csv")

# ===================== 🧹 TAB 11: Final Column Order Template Reorder =====================

with tab11:
    st.title("🧾 Final Column Order Template Reorder")

    uploaded_final_csv = st.file_uploader("📤 Please upload your .csv file...", type=["csv"], key="finaltab")

    if uploaded_final_csv:
        df = pd.read_csv(uploaded_final_csv)

        fixed_column = "{{custom_id}}"

        template_str = """{{storytitle}}\t{{pagetitle}}\t{{uuid}}\t{{urlslug}}\t{{canurl}}\t{{canurl 1}} \t{{publishedtime}}\t{{modifiedtime}}\t{{metakeywords}}\t{{metadescription}}\t{{s2paragraph1}}\t{{s3paragraph1}}\t{{s4paragraph1}}\t{{s5paragraph1}}\t{{s6paragraph1}}\t{{s7paragraph1}}\t{{s8paragraph1}}\t{{s9paragraph1}}\t{{s1alt1}}\t{{s2alt1}}\t{{s3alt1}}\t{{s4alt1}}\t{{s5alt1}}\t{{s6alt1}}\t{{s7alt1}}\t{{s8alt1}}\t{{s9alt1}}\t{{hookline}}\t{{potraightcoverurl}}\t{{landscapecoverurl}}\t{{squarecoverurl}}\t{{socialthumbnailcoverurl}}\t{{s1image1}}\t{{s2image1}}\t{{s3image1}}\t{{s4image1}}\t{{s5image1}}\t{{s6image1}}\t{{s7image1}}\t{{s8image1}}\t{{s9image1}}\t{{s11btntext}}\t{{s11btnlink}}\t{{lang}}\t{{prevstorytitle}}\t{{prevstorylink}}\t{{nextstorytitle}}\t{{nextstorylink}}\t{{user}}\t{{userprofileurl}}\t{{storygeneratorname}}\t{{contenttype}}\t{{storygeneratorversion}}\t{{sitename}}\t{{generatorplatform}}\t{{sitelogo96x96}}\t{{person}}\t{{sitelogo32x32}}\t{{sitelogo192x192}}\t{{sitelogo144x144}}\t{{sitelogo92x92}}\t{{sitelogo180x180}}\t{{publisher}}\t{{publisherlogosrc}}\t{{gtagid}}\t{{organization}}\t{{publisherlogoalt}}\t{{s10video1}}\t{{s10alt1}}\t{{videoscreenshot}}\t{{writername}}\t{{s10caption1}}\t{{s11paragraph1}}\t{{nextstoryimage}}\t{{nextstoryimagealt}}"""

        template_columns = [col.strip() for col in template_str.split("\t")]

        missing_cols = []
        for col in template_columns:
            if col not in df.columns:
                df[col] = ""
                missing_cols.append(col)

        if fixed_column not in df.columns:
            st.error("❌ Column {{custom_id}} is missing from the uploaded CSV!")
        else:
            final_order = [fixed_column] + template_columns
            df = df[final_order]

            st.subheader("✅ Final Reordered CSV Preview")
            st.dataframe(df.head())

            output_filename = f"Final-Datasheet_{int(time.time())}.csv"
            st.download_button("📥 Download Final Reordered CSV", data=df.to_csv(index=False), file_name=output_filename, mime="text/csv")

            if missing_cols:
                st.warning("⚠️ The following columns were missing and have been filled with blanks:")
                for col in missing_cols:
                    st.markdown(f"- ❌ `{col}`")
            else:
                st.success("✅ All template columns (besides {{custom_id}}) were present.")
