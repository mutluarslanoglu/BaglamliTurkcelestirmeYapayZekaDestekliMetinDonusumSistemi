
dosyanın üzerinden cmd aç
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Aktif olunca satır başında (.venv) görürsün.
pip install -r requirements.txt

(LOQda bu alttaki ikisiniz yapsan yeterli-kurumlumdan sonra hep bu artık)
uvicorn app:app --reload


web dosyasında index html aç 