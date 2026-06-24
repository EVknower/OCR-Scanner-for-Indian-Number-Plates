import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import database_manager as db
from export_excel import export_all, export_today
from config import CAMERA_SOURCE

st.set_page_config(page_title="ANPR Dashboard", layout="wide")
st.title("🚗 Real-Time ANPR System")

tab1, tab2, tab3 = st.tabs(["📷 Live Feed", "📋 Records", "🔍 Search"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Camera Feed")
        run = st.toggle("Start Camera")
        frame_placeholder = st.empty()
    with col2:
        st.subheader("Last Detected")
        plate_placeholder = st.empty()
        conf_placeholder = st.empty()
        count_placeholder = st.empty()

    if run:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from main import ANPRPipeline
        pipeline = ANPRPipeline()
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_count = 0
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not available.")
                break
            frame = cv2.flip(frame, 1)
            frame_count += 1
            annotated, detections = pipeline.process_frame(frame, frame_count)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, width="stretch")
            if detections:
                last = detections[-1]
                plate_placeholder.metric("Plate", last["plate"])
                conf_placeholder.metric("Confidence", f"{last['confidence']*100:.1f}%")
            today = db.fetch_today()
            count_placeholder.metric("Vehicles Today", len(today))
        cap.release()

with tab2:
    st.subheader("Vehicle Log")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Export All → CSV"):
            path = export_all()
            st.success(f"Saved: {path}")
    with col_b:
        if st.button("Export Today → CSV"):
            path = export_today()
            st.success(f"Saved: {path}")
    with col_c:
        if st.button("Refresh"):
            st.rerun()

    rows = db.fetch_all()
    if rows:
        df = pd.DataFrame(rows, columns=["Plate", "Timestamp", "Confidence", "Snapshot"])
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.1f}%" if x else "-")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No records yet. Start the camera to begin logging.")

with tab3:
    st.subheader("Search by Plate Number")
    query = st.text_input("Enter plate number (or partial)", placeholder="e.g. UP32")
    if query:
        results = db.search(query.upper())
        if results:
            df = pd.DataFrame(results, columns=["Plate", "Timestamp", "Confidence", "Snapshot"])
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.1f}%" if x else "-")
            st.dataframe(df, use_container_width=True)
            for row in results:
                if row[3] and os.path.exists(row[3]):
                    st.image(row[3], caption=f"{row[0]} @ {row[1]}", width=300)
        else:
            st.warning("No results found.")
