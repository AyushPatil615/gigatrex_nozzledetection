"""
Start FastAPI Backend & Open ngrok Public Tunnel
"""

import sys
import os
import time
import subprocess
from pyngrok import ngrok, conf

def main():
    # If user provided an authtoken in environment or as argument
    token = os.environ.get("NGROK_AUTHTOKEN")
    if len(sys.argv) > 1:
        token = sys.argv[1]

    if token:
        print(f"[INFO] Setting ngrok authtoken...")
        ngrok.set_auth_token(token)
    else:
        print("[INFO] No authtoken passed. If you have an ngrok account, run: python start_tunnel.py YOUR_AUTHTOKEN")

    PORT = 8000

    print("\n" + "="*65)
    print("  🚀 Starting NozzleVision FastAPI Server + ngrok Tunnel")
    print("="*65)

    try:
        # Open ngrok HTTP tunnel
        public_url = ngrok.connect(PORT).public_url
        print(f"\n  🌐 PUBLIC NGROK URL (Paste this into Vercel Frontend):")
        print(f"     👉 {public_url}\n")
        print("="*65 + "\n")
    except Exception as e:
        print(f"[WARNING] ngrok connection note: {e}")
        print("You can get a free authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")
        print(f"Server is still running locally on http://127.0.0.1:{PORT}\n")

    # Start Uvicorn
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=PORT, reload=False)

if __name__ == "__main__":
    main()
