@echo off
echo Activating yj_pytorch environment...
call C:\Users\jour\anaconda3\Scripts\activate.bat yj_pytorch

echo Generating v2 LLM embeddings...
python generate_local_embeddings_v2.py

echo Done!
pause
