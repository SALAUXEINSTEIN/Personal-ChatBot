---
title: chatbot_project
app_file: chatbot_project/app/gradio_app.py
sdk: gradio
sdk_version: 6.20.0
---
# Personalised Transformer Chatbot

This repository contains the reference implementation for a personalised dialogue system built with a transformer backbone and Gradio demo interface.

## Quick Start

1. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Run the quick demo without downloading large checkpoints:

```powershell
.\.venv\Scripts\python.exe -u -m chatbot_project.app.gradio_app --quick_demo
```

3. Open the UI in your browser at:

```
http://127.0.0.1:7861
```

## Full Model Run

To run the real model, you need local checkpoint directories:

- `checkpoints/stage1_backbone`
- `checkpoints/stage2_full_system`

Then launch with:

```powershell
.\.venv\Scripts\python.exe -u -m chatbot_project.app.gradio_app --backbone_checkpoint checkpoints/stage1_backbone --stage2_checkpoint checkpoints/stage2_full_system
```

If those folders are missing, the app will try to load them from Hugging Face and fail unless they are valid model repositories or you have authentication set up.

## Docker

Build the image:

```powershell
docker build -t chatbot_project .
```

Run the quick demo:

```powershell
docker run --rm -p 7860:7860 chatbot_project
```

Run the full model with local checkpoints mounted:

```powershell
docker run --rm -p 7860:7860 -v %cd%/checkpoints:/app/checkpoints chatbot_project --backbone_checkpoint checkpoints/stage1_backbone --stage2_checkpoint checkpoints/stage2_full_system
```

## GitHub Deployment

1. Initialize git in the repo root:

```powershell
git init
git add .
git commit -m "Initial chatbot project setup"
```

2. Create a GitHub repository and add it as remote:

```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

3. Use the built-in GitHub Actions workflow in `.github/workflows/python-app.yml` for package validation.
