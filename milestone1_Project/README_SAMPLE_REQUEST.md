# Sample Requests

## cURL
```bash
curl -X POST "http://localhost:8000/cleanup" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@sample_input.jpg"
```

## n8n HTTP Request node
- Method: POST
- URL: http://YOUR_SERVER:8000/cleanup
- Body Content Type: form-data
- Key: file (type: File)
- Value: upload or binary from previous node
