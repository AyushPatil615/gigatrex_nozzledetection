"""
NozzleVision AI - Main CLI & Entry Point
"""

import sys
import os

if __name__ == "__main__":
    print("\n" + "="*50)
    print("      NOZZLEVISION AI — TRAY INSPECTOR")
    print("="*50)
    print("Starting Web Interface for Panel Presentation...")
    
    from app import create_app
    demo = create_app()
    # share=True creates a public URL accessible from any device (phone, panel laptop)
    demo.launch(share=True, inbrowser=True)
