FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN curl -fsSL https://ollama.com/install.sh | bash

COPY . .

EXPOSE 8501
# CMD ["python","-m","streamlit", "run","./src/app.py"]
# CMD /bin/bash -c "ollama serve & sleep 3 & ollama pull gemma3 && python -m streamlit run ./src/app.py --server.address=0.0.0.0 --server.port=8501"
CMD ["bash", "-c", "ollama serve & sleep 3 && ollama pull gemma3 && streamlit run ./src/app.py" ]
