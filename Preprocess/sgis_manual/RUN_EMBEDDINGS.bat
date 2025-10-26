@echo off
echo ================================================================================
echo SGIS Local Embeddings Generation
echo ================================================================================
echo.
echo This will generate LLM embeddings for 28,274 SGIS prompts
echo Estimated time: 30-45 minutes with RTX 4070 GPU
echo.
echo IMPORTANT: Make sure you've set your HF_TOKEN first!
echo   In PowerShell: $env:HF_TOKEN="your_token_here"
echo   In CMD: set HF_TOKEN=your_token_here
echo.
pause

echo.
echo Starting embedding generation...
echo.

python generate_local_embeddings.py

echo.
echo ================================================================================
echo.
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS! Embeddings generated successfully.
    echo Output file: sgis_local_llm_embeddings.csv
) else (
    echo ERROR! Something went wrong. Check the error messages above.
)
echo.
pause
