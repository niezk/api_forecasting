import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# Load the service account from environment variable
service_account_info = json.loads(os.environ["FIREBASE_CREDENTIALS"])
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# cred_path = os.path.join(BASE_DIR, "ServiceAccount.json")
# service_account_info = json.loads(cred_path)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

# Export Firestore client
firebase_db = firestore.client()
