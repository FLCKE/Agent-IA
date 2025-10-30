# Reactive AI Agent

## Description

Agent réactif utilisant Ollama.

## Installation

### Local

ollama run mistral "Bonjour !"
py -m streamlit run ./src/app.py

### Docker

docker build -t ai-agent-lab .
docker run -p 8501:8501 ai-agent-lab

## Fonctionnement

Input → LLM → Output

## Auteur

FRANCISCO Louis-Carlos – Ydays 2025
