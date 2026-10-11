# SIH26084 - Convective scale nowcasting for Thunderstorms, Hail & Cloudbursts (06 hr)

Dedicated project codebase for **Smart India Hackathon 2026** problem statement `SIH26084`.

## 🚀 Execution Instructions

### Instant UI Browser View (Zero Dependencies)
Double-click `index.html` or serve via Python:
```bash
python -m http.server 8080
```
Visit: [http://localhost:8080](http://localhost:8080)

### Full FastAPI Backend
```bash
pip install -r requirements.txt
python app.py
```
- API Endpoint: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### Automated Unit Testing
```bash
pytest test_app.py -v
```

### Docker Containerization
```bash
docker-compose up --build
```
