# Alzheimer’s MRI & RAG system

A project that combines:
- **3D MRI visualization** from NIfTI files (`.nii/.nii.gz`)
- **2D MRI slice classification** using a deep learning model
- **Biomarker exploration** CSF + plasma markers from ADNI analyses
- **RAG Q&A over Alzheimer related documents** using LangChain + LangGraph + FAISS
  
> This project is still under development will include better RAG strategies adaptive retrieval, validation, and domain-specific routing.

## Features

### 1) 3D MRI Viewer (NIfTI)
Upload a brain MRI scan (`.nii` / `.nii.gz`) and explore it interactively using Plotly ( I have given a  folder where you can access few files):
- Axial / Coronal / Sagittal views
- Slice-by-slice animation + slider
- Downsampling for lower memory usage (`zoom(..., 0.5)`)

### 2) 2D MRI Prediction
Upload a single **2D MRI slice** (`.png/.jpg`) and get :
- Predicted class: **CN / EMCI / LMCI / AD**
**Label meanings**
- **CN**: Cognitive Normal  
- **EMCI**: Early Mild Cognitive Impairment  
- **LMCI**: Late Mild Cognitive Impairment  
- **AD**: Alzheimer’s Disease  
- Per-class confidence scores
( I have given a  folder where you can access few files)

### 3) Biomarker Prediction (In Progress)
A biomarker-focused section (currently only exploratory) showing distributions of key markers such as:
- CSF: **Aβ42, Aβ40, Tau, pTau**
- Plasma: **GFAP, NfL, pTau217, Aβ42/40 ratio**

Includes violin plots and interpretation notes (work in progress → modeling + stats coming next).

### 4) RAG Q&A over Alzheimer Documents (LangChain + LangGraph)
Ask questions over a local collection of Alzheimer docs:
- Document ingestion → chunking
- Embedding → FAISS vector store
- Retrieval → LLM response